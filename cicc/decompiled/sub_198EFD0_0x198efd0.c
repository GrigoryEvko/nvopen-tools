// Function: sub_198EFD0
// Address: 0x198efd0
//
void __fastcall sub_198EFD0(__int64 *src, __int64 *a2, __int64 a3)
{
  __int64 *v3; // r12
  __int64 v5; // rbx
  __int64 v6; // r10
  __int64 *v7; // r8
  unsigned int v8; // r14d
  __int64 v9; // r13
  int v10; // ecx
  unsigned int v11; // esi
  __int64 *v12; // rax
  __int64 v13; // r12
  __int64 v14; // rdi
  char v15; // dl
  __int64 v16; // r9
  int v17; // ecx
  unsigned int v18; // r11d
  __int64 *v19; // rax
  __int64 v20; // rsi
  int v21; // r11d
  __int64 v22; // rcx
  __int64 v23; // rcx
  __int64 v24; // rcx
  __int64 v25; // rcx
  int v26; // eax
  int v27; // eax
  int v28; // r12d
  int v29; // [rsp+Ch] [rbp-54h]
  __int64 *v30; // [rsp+10h] [rbp-50h]
  __int64 v32[7]; // [rsp+28h] [rbp-38h] BYREF

  v32[0] = a3;
  if ( src != a2 )
  {
    v3 = src + 1;
    if ( a2 != src + 1 )
    {
      do
      {
        while ( 1 )
        {
          v5 = *v3;
          if ( !sub_198ECB0(v32, *v3, *src) )
            break;
          if ( src != v3 )
            memmove(src + 1, src, (char *)v3 - (char *)src);
          *src = v5;
          if ( a2 == ++v3 )
            return;
        }
        v6 = v32[0];
        v30 = v3;
        v7 = v3;
        v8 = ((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4);
        v9 = v32[0] + 16;
        while ( 1 )
        {
          v14 = *(v7 - 1);
          v15 = *(_BYTE *)(v6 + 8) & 1;
          if ( v15 )
          {
            v16 = v9;
            v17 = 15;
          }
          else
          {
            v23 = *(unsigned int *)(v6 + 24);
            v16 = *(_QWORD *)(v6 + 16);
            if ( !(_DWORD)v23 )
              goto LABEL_27;
            v17 = v23 - 1;
          }
          v18 = v8 & v17;
          v19 = (__int64 *)(v16 + 16LL * (v8 & v17));
          v20 = *v19;
          if ( v5 == *v19 )
            goto LABEL_16;
          v27 = 1;
          while ( v20 != -8 )
          {
            v28 = v27 + 1;
            v18 = v17 & (v27 + v18);
            v19 = (__int64 *)(v16 + 16LL * v18);
            v20 = *v19;
            if ( v5 == *v19 )
              goto LABEL_16;
            v27 = v28;
          }
          if ( v15 )
          {
            v25 = 256;
            goto LABEL_28;
          }
          v23 = *(unsigned int *)(v6 + 24);
LABEL_27:
          v25 = 16 * v23;
LABEL_28:
          v19 = (__int64 *)(v16 + v25);
LABEL_16:
          v21 = *((_DWORD *)v19 + 2);
          if ( v15 )
          {
            v10 = 15;
          }
          else
          {
            v22 = *(unsigned int *)(v6 + 24);
            if ( !(_DWORD)v22 )
              goto LABEL_22;
            v10 = v22 - 1;
          }
          v11 = v10 & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
          v12 = (__int64 *)(v16 + 16LL * v11);
          v13 = *v12;
          if ( v14 != *v12 )
            break;
LABEL_11:
          if ( v21 >= *((_DWORD *)v12 + 2) )
            goto LABEL_24;
LABEL_12:
          *v7-- = v14;
        }
        v26 = 1;
        while ( v13 != -8 )
        {
          v11 = v10 & (v26 + v11);
          v29 = v26 + 1;
          v12 = (__int64 *)(v16 + 16LL * v11);
          v13 = *v12;
          if ( v14 == *v12 )
            goto LABEL_11;
          v26 = v29;
        }
        if ( v15 )
        {
          v24 = 256;
        }
        else
        {
          v22 = *(unsigned int *)(v6 + 24);
LABEL_22:
          v24 = 16 * v22;
        }
        if ( v21 < *(_DWORD *)(v16 + v24 + 8) )
          goto LABEL_12;
LABEL_24:
        *v7 = v5;
        v3 = v30 + 1;
      }
      while ( a2 != v30 + 1 );
    }
  }
}
