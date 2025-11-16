// Function: sub_2AF1190
// Address: 0x2af1190
//
void __fastcall sub_2AF1190(__int64 a1, unsigned __int64 a2, unsigned __int64 a3)
{
  unsigned __int64 v3; // rax
  unsigned int v4; // r13d
  unsigned __int64 v5; // r12
  unsigned int v7; // edx
  __int64 v8; // r15
  __int64 v9; // r9
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 *v12; // rcx
  unsigned __int64 v13; // rsi
  __int64 v14; // r8
  int v15; // eax
  __int64 *v16; // rdx
  unsigned __int64 v17; // r15
  __int64 v18; // rax
  __int64 v19; // rdi
  char *v20; // r15
  __int64 v21; // rdi
  char v22; // r15
  unsigned __int64 v23; // r12
  unsigned __int64 v24; // [rsp+28h] [rbp-68h] BYREF
  unsigned __int64 v25; // [rsp+30h] [rbp-60h]
  _QWORD v26[2]; // [rsp+38h] [rbp-58h] BYREF
  __int64 v27; // [rsp+48h] [rbp-48h]
  unsigned __int64 v28; // [rsp+50h] [rbp-40h] BYREF
  unsigned int v29; // [rsp+58h] [rbp-38h]
  char v30; // [rsp+5Ch] [rbp-34h]

  v3 = HIDWORD(a2);
  v4 = 2 * a3;
  v5 = HIDWORD(a3);
  v25 = a2;
  v7 = a2;
  while ( (!(_BYTE)v3 || (_BYTE)v5) && v4 > v7 )
  {
    LODWORD(v25) = v7;
    BYTE4(v25) = v3;
    v29 = v4;
    v28 = v25;
    v30 = v5;
    sub_2AEE460((__int64 *)&v24, (void **)a1, (__int64)&v28);
    v8 = v24;
    if ( v24 )
    {
      if ( *(_DWORD *)(v24 + 88) == 1 && (v18 = *(_QWORD *)(v24 + 80), !*(_BYTE *)(v18 + 4)) && *(_DWORD *)v18 == 1 )
      {
        sub_2C3ACB0(v24);
      }
      else
      {
        sub_2C34E00(v24, *(_QWORD *)(a1 + 48) + 16LL);
        if ( LOBYTE(qword_500D260[17]) )
          sub_2C4B640(v8);
        sub_2C3ACB0(v24);
        v10 = *(_QWORD *)(a1 + 48);
        if ( *(_BYTE *)(v10 + 108) && *(_DWORD *)(v10 + 100) == 5 )
        {
          v27 = *(_QWORD *)(v10 + 116);
          v26[1] = v27;
          v26[0] = v27;
          v21 = v24;
          v22 = sub_2C3BCB0(v24, v26);
          if ( LOBYTE(qword_500D260[17]) )
            sub_2C4B640(v21);
          if ( !v22 )
          {
            v23 = v24;
            if ( v24 )
            {
              sub_2BF1F00(v24);
              j_j___libc_free_0(v23);
            }
            return;
          }
        }
      }
      v11 = *(unsigned int *)(a1 + 96);
      v12 = (__int64 *)&v24;
      v13 = *(_QWORD *)(a1 + 88);
      v14 = v11 + 1;
      v15 = *(_DWORD *)(a1 + 96);
      if ( v11 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 100) )
      {
        v19 = a1 + 88;
        if ( v13 > (unsigned __int64)&v24 || (unsigned __int64)&v24 >= v13 + 8 * v11 )
        {
          sub_2AC3D10(v19, v11 + 1, v11, (__int64)&v24, v14, v9);
          v11 = *(unsigned int *)(a1 + 96);
          v13 = *(_QWORD *)(a1 + 88);
          v12 = (__int64 *)&v24;
          v15 = *(_DWORD *)(a1 + 96);
        }
        else
        {
          v20 = (char *)&v24 - v13;
          sub_2AC3D10(v19, v11 + 1, v11, (__int64)&v24, v14, v9);
          v13 = *(_QWORD *)(a1 + 88);
          v11 = *(unsigned int *)(a1 + 96);
          v12 = (__int64 *)&v20[v13];
          v15 = *(_DWORD *)(a1 + 96);
        }
      }
      v16 = (__int64 *)(v13 + 8 * v11);
      if ( v16 )
      {
        *v16 = *v12;
        *v12 = 0;
        v15 = *(_DWORD *)(a1 + 96);
      }
      v17 = v24;
      *(_DWORD *)(a1 + 96) = v15 + 1;
      if ( v17 )
      {
        sub_2BF1F00(v17);
        j_j___libc_free_0(v17);
      }
    }
    v7 = v29;
    LOBYTE(v3) = v30;
  }
}
