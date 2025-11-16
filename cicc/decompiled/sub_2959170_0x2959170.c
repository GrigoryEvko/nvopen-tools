// Function: sub_2959170
// Address: 0x2959170
//
void __fastcall sub_2959170(__int64 *src, __int64 *a2, __int64 a3)
{
  __int64 *v3; // r13
  __int64 v6; // r15
  __int64 *v7; // rcx
  __int64 v8; // r12
  __int64 *v9; // r8
  __int64 v10; // rdi
  unsigned int v11; // r10d
  char v12; // dl
  __int64 v13; // r9
  int v14; // esi
  unsigned int v15; // r11d
  __int64 *v16; // rax
  __int64 v17; // rcx
  _QWORD *v18; // rcx
  unsigned int i; // eax
  unsigned int v20; // ecx
  __int64 *v21; // rdx
  __int64 v22; // r11
  unsigned int v23; // ecx
  _QWORD *j; // rdx
  int v25; // esi
  int v26; // edx
  int v27; // eax
  int v28; // [rsp+0h] [rbp-40h]
  int v29; // [rsp+0h] [rbp-40h]

  if ( src != a2 )
  {
    v3 = src + 1;
    if ( a2 != src + 1 )
    {
      v6 = a3 + 16;
      do
      {
        while ( 1 )
        {
          v8 = *v3;
          if ( !sub_2959010(a3, *v3, *src) )
            break;
          v7 = v3 + 1;
          if ( src != v3 )
          {
            memmove(src + 1, src, (char *)v3 - (char *)src);
            v7 = v3 + 1;
          }
          *src = v8;
          v3 = v7;
          if ( a2 == v7 )
            return;
        }
        v9 = v3;
        v10 = *(v3 - 1);
        v11 = ((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4);
        v12 = *(_BYTE *)(a3 + 8) & 1;
        if ( v12 )
        {
LABEL_9:
          v13 = v6;
          v14 = 15;
          goto LABEL_10;
        }
        while ( 1 )
        {
          v25 = *(_DWORD *)(a3 + 24);
          v13 = *(_QWORD *)(a3 + 16);
          if ( !v25 )
            BUG();
          v14 = v25 - 1;
LABEL_10:
          v15 = v14 & v11;
          v16 = (__int64 *)(v13 + 16LL * (v14 & v11));
          v17 = *v16;
          if ( v8 != *v16 )
          {
            v27 = 1;
            while ( v17 != -4096 )
            {
              v15 = v14 & (v27 + v15);
              v29 = v27 + 1;
              v16 = (__int64 *)(v13 + 16LL * v15);
              v17 = *v16;
              if ( v8 == *v16 )
                goto LABEL_11;
              v27 = v29;
            }
LABEL_33:
            BUG();
          }
LABEL_11:
          v18 = *(_QWORD **)v16[1];
          for ( i = 1; v18; ++i )
            v18 = (_QWORD *)*v18;
          if ( !v12 && !*(_DWORD *)(a3 + 24) )
            goto LABEL_33;
          v20 = v14 & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
          v21 = (__int64 *)(v13 + 16LL * v20);
          v22 = *v21;
          if ( *v21 != v10 )
          {
            v26 = 1;
            while ( v22 != -4096 )
            {
              v20 = v14 & (v26 + v20);
              v28 = v26 + 1;
              v21 = (__int64 *)(v13 + 16LL * v20);
              v22 = *v21;
              if ( v10 == *v21 )
                goto LABEL_16;
              v26 = v28;
            }
            goto LABEL_33;
          }
LABEL_16:
          v23 = 1;
          for ( j = *(_QWORD **)v21[1]; j; ++v23 )
            j = (_QWORD *)*j;
          if ( v23 <= i )
            break;
          *v9-- = v10;
          v10 = *(v9 - 1);
          v12 = *(_BYTE *)(a3 + 8) & 1;
          if ( v12 )
            goto LABEL_9;
        }
        *v9 = v8;
        ++v3;
      }
      while ( a2 != v3 );
    }
  }
}
