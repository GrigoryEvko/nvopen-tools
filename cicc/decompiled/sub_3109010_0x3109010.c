// Function: sub_3109010
// Address: 0x3109010
//
void __fastcall sub_3109010(__int64 a1, __int64 a2, __int64 a3, unsigned __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rbx
  __int64 v8; // r12
  _QWORD *v9; // rax
  __int64 v10; // rdx
  unsigned __int64 *v11; // r15
  unsigned __int64 v12; // rsi
  int v13; // eax
  int v14; // eax
  bool v15; // cc
  __int64 v16; // rdi
  char *v17; // r15
  unsigned __int64 v18; // [rsp+0h] [rbp-50h] BYREF
  unsigned __int64 v19; // [rsp+8h] [rbp-48h] BYREF
  unsigned int v20; // [rsp+10h] [rbp-40h]

  v6 = *(_QWORD *)(a2 + 16);
  if ( v6 )
  {
    v8 = a1 + 240;
    while ( 1 )
    {
      if ( !*(_BYTE *)(a1 + 268) )
        goto LABEL_10;
      v9 = *(_QWORD **)(a1 + 248);
      a4 = *(unsigned int *)(a1 + 260);
      a3 = (__int64)&v9[a4];
      if ( v9 != (_QWORD *)a3 )
      {
        while ( *v9 != v6 )
        {
          if ( (_QWORD *)a3 == ++v9 )
            goto LABEL_20;
        }
        goto LABEL_8;
      }
LABEL_20:
      if ( (unsigned int)a4 < *(_DWORD *)(a1 + 256) )
      {
        *(_DWORD *)(a1 + 260) = a4 + 1;
        *(_QWORD *)a3 = v6;
        ++*(_QWORD *)(a1 + 240);
LABEL_11:
        v18 = (4LL * *(unsigned __int8 *)(a1 + 344)) | v6 & 0xFFFFFFFFFFFFFFFBLL;
        v20 = *(_DWORD *)(a1 + 360);
        if ( v20 > 0x40 )
          sub_C43780((__int64)&v19, (const void **)(a1 + 352));
        else
          v19 = *(_QWORD *)(a1 + 352);
        v10 = *(unsigned int *)(a1 + 40);
        v11 = &v18;
        a4 = *(_QWORD *)(a1 + 32);
        v12 = v10 + 1;
        v13 = *(_DWORD *)(a1 + 40);
        if ( v10 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 44) )
        {
          v16 = a1 + 32;
          if ( a4 > (unsigned __int64)&v18 || (unsigned __int64)&v18 >= a4 + 24 * v10 )
          {
            sub_3108F20(v16, v12, v10, a4, a5, a6);
            v10 = *(unsigned int *)(a1 + 40);
            a4 = *(_QWORD *)(a1 + 32);
            v13 = *(_DWORD *)(a1 + 40);
          }
          else
          {
            v17 = (char *)&v18 - a4;
            sub_3108F20(v16, v12, v10, a4, a5, a6);
            a4 = *(_QWORD *)(a1 + 32);
            v10 = *(unsigned int *)(a1 + 40);
            v11 = (unsigned __int64 *)&v17[a4];
            v13 = *(_DWORD *)(a1 + 40);
          }
        }
        a3 = a4 + 24 * v10;
        if ( a3 )
        {
          *(_QWORD *)a3 = *v11;
          v14 = *((_DWORD *)v11 + 4);
          *((_DWORD *)v11 + 4) = 0;
          *(_DWORD *)(a3 + 16) = v14;
          *(_QWORD *)(a3 + 8) = v11[1];
          v13 = *(_DWORD *)(a1 + 40);
        }
        v15 = v20 <= 0x40;
        *(_DWORD *)(a1 + 40) = v13 + 1;
        if ( v15 || !v19 )
          goto LABEL_8;
        j_j___libc_free_0_0(v19);
        v6 = *(_QWORD *)(v6 + 8);
        if ( !v6 )
          return;
      }
      else
      {
LABEL_10:
        sub_C8CC70(v8, v6, a3, a4, a5, a6);
        if ( (_BYTE)a3 )
          goto LABEL_11;
LABEL_8:
        v6 = *(_QWORD *)(v6 + 8);
        if ( !v6 )
          return;
      }
    }
  }
}
