// Function: sub_30B1F70
// Address: 0x30b1f70
//
__int64 __fastcall sub_30B1F70(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  char *v3; // rbx
  char *v4; // r10
  __int64 v5; // r11
  __int64 v6; // r12
  __int64 v7; // r13
  _QWORD **v8; // rbx
  __int64 v9; // rdx
  _QWORD **i; // r15
  _QWORD *v11; // r8
  __int64 *v12; // rbx
  __int64 *j; // r15
  __int64 v14; // rdx
  int v15; // eax
  __int64 v16; // rdi
  int v17; // ecx
  unsigned int v18; // eax
  __int64 *v19; // rsi
  __int64 v20; // r8
  __int64 v21; // rax
  _QWORD *v22; // rdi
  __int64 v23; // rsi
  _QWORD *v24; // rax
  int v25; // r8d
  int v26; // esi
  int v27; // r9d
  __int64 v28; // rax
  __int64 v29; // rdx
  __int64 *v30; // rdi
  _QWORD *v32; // [rsp+0h] [rbp-E0h]
  char *dest; // [rsp+20h] [rbp-C0h]
  __int64 v36; // [rsp+30h] [rbp-B0h]
  char *v37; // [rsp+38h] [rbp-A8h]
  __int64 v38; // [rsp+48h] [rbp-98h] BYREF
  __int64 *v39; // [rsp+50h] [rbp-90h] BYREF
  __int64 v40; // [rsp+58h] [rbp-88h]
  _BYTE v41[128]; // [rsp+60h] [rbp-80h] BYREF

  v2 = *(unsigned int *)(a1 + 8);
  v3 = *(char **)a1;
  v36 = *(_QWORD *)a1 + 8 * v2;
  dest = sub_30B0410(v3, &v3[8 * v2], a2);
  if ( v4 == dest )
    return 0;
  v37 = v3;
  v6 = v5 + 8;
  v39 = (__int64 *)v41;
  v40 = 0xA00000000LL;
  if ( v3 != v4 )
  {
    do
    {
      v7 = *(_QWORD *)v37;
      if ( v6 != *(_QWORD *)v37 + 8LL )
      {
        v8 = *(_QWORD ***)(v7 + 40);
        v9 = (unsigned int)v40;
        for ( i = &v8[*(unsigned int *)(v7 + 48)]; i != v8; LODWORD(v40) = v40 + 1 )
        {
          while ( 1 )
          {
            v11 = *v8;
            if ( v6 == **v8 + 8LL )
              break;
            if ( i == ++v8 )
              goto LABEL_13;
          }
          if ( v9 + 1 > (unsigned __int64)HIDWORD(v40) )
          {
            v32 = *v8;
            sub_C8D5F0((__int64)&v39, v41, v9 + 1, 8u, (__int64)v11, v9 + 1);
            v9 = (unsigned int)v40;
            v11 = v32;
          }
          ++v8;
          v39[v9] = (__int64)v11;
          v9 = (unsigned int)(v40 + 1);
        }
LABEL_13:
        v12 = &v39[v9];
        for ( j = v39; v12 != j; ++j )
        {
          v14 = *j;
          v15 = *(_DWORD *)(v7 + 32);
          v16 = *(_QWORD *)(v7 + 16);
          v38 = *j;
          if ( v15 )
          {
            v17 = v15 - 1;
            v18 = (v15 - 1) & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
            v19 = (__int64 *)(v16 + 8LL * v18);
            v20 = *v19;
            if ( v14 == *v19 )
            {
LABEL_16:
              *v19 = -8192;
              v21 = *(unsigned int *)(v7 + 48);
              --*(_DWORD *)(v7 + 24);
              v22 = *(_QWORD **)(v7 + 40);
              ++*(_DWORD *)(v7 + 28);
              v23 = (__int64)&v22[v21];
              v24 = sub_30B02A0(v22, v23, &v38);
              if ( v24 + 1 != (_QWORD *)v23 )
              {
                memmove(v24, v24 + 1, v23 - (_QWORD)(v24 + 1));
                v25 = *(_DWORD *)(v7 + 48);
              }
              *(_DWORD *)(v7 + 48) = v25 - 1;
            }
            else
            {
              v26 = 1;
              while ( v20 != -4096 )
              {
                v27 = v26 + 1;
                v18 = v17 & (v26 + v18);
                v19 = (__int64 *)(v16 + 8LL * v18);
                v20 = *v19;
                if ( v14 == *v19 )
                  goto LABEL_16;
                v26 = v27;
              }
            }
          }
        }
        LODWORD(v40) = 0;
      }
      v37 += 8;
    }
    while ( (char *)v36 != v37 );
  }
  sub_30B1DB0(v6);
  *(_DWORD *)(a2 + 48) = 0;
  v28 = *(unsigned int *)(a1 + 8);
  v29 = *(_QWORD *)a1 + 8 * v28;
  if ( (char *)v29 != dest + 8 )
  {
    memmove(dest, dest + 8, v29 - (_QWORD)(dest + 8));
    LODWORD(v28) = *(_DWORD *)(a1 + 8);
  }
  v30 = v39;
  *(_DWORD *)(a1 + 8) = v28 - 1;
  if ( v30 != (__int64 *)v41 )
    _libc_free((unsigned __int64)v30);
  return 1;
}
