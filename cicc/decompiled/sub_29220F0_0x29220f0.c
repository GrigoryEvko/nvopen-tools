// Function: sub_29220F0
// Address: 0x29220f0
//
char __fastcall sub_29220F0(__int64 a1, __int64 a2)
{
  unsigned __int8 *v3; // r13
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rdx
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // rdx
  char *v10; // rbx
  unsigned __int64 v11; // rcx
  unsigned __int64 v12; // rsi
  unsigned __int64 *v13; // rdi
  unsigned __int64 v14; // rax
  __int64 v15; // rdi
  char *v16; // rbx
  _QWORD v18[2]; // [rsp+0h] [rbp-40h] BYREF
  unsigned __int8 *v19; // [rsp+10h] [rbp-30h]

  v3 = *(unsigned __int8 **)a2;
  v4 = sub_ACADE0(*(__int64 ***)(*(_QWORD *)a2 + 8LL));
  if ( *(_QWORD *)a2 )
  {
    v5 = *(_QWORD *)(a2 + 8);
    **(_QWORD **)(a2 + 16) = v5;
    if ( v5 )
      *(_QWORD *)(v5 + 16) = *(_QWORD *)(a2 + 16);
  }
  *(_QWORD *)a2 = v4;
  if ( v4 )
  {
    v6 = *(_QWORD *)(v4 + 16);
    *(_QWORD *)(a2 + 8) = v6;
    if ( v6 )
      *(_QWORD *)(v6 + 16) = a2 + 8;
    *(_QWORD *)(a2 + 16) = v4 + 16;
    *(_QWORD *)(v4 + 16) = a2;
  }
  if ( *v3 > 0x1Cu )
  {
    LOBYTE(v4) = sub_F50EE0(v3, 0);
    if ( (_BYTE)v4 )
    {
      v18[0] = 4;
      v18[1] = 0;
      v19 = v3;
      if ( v3 != (unsigned __int8 *)-8192LL && v3 != (unsigned __int8 *)-4096LL )
        sub_BD73F0((__int64)v18);
      v9 = *(unsigned int *)(a1 + 224);
      v10 = (char *)v18;
      v11 = *(_QWORD *)(a1 + 216);
      v12 = v9 + 1;
      LODWORD(v4) = *(_DWORD *)(a1 + 224);
      if ( v9 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 228) )
      {
        v15 = a1 + 216;
        if ( v11 > (unsigned __int64)v18 || (unsigned __int64)v18 >= v11 + 24 * v9 )
        {
          sub_D6B130(v15, v12, v9, v11, v7, v8);
          v9 = *(unsigned int *)(a1 + 224);
          v11 = *(_QWORD *)(a1 + 216);
          LODWORD(v4) = *(_DWORD *)(a1 + 224);
        }
        else
        {
          v16 = (char *)v18 - v11;
          sub_D6B130(v15, v12, v9, v11, v7, v8);
          v11 = *(_QWORD *)(a1 + 216);
          v9 = *(unsigned int *)(a1 + 224);
          v10 = &v16[v11];
          LODWORD(v4) = *(_DWORD *)(a1 + 224);
        }
      }
      v13 = (unsigned __int64 *)(v11 + 24 * v9);
      if ( v13 )
      {
        *v13 = 4;
        v14 = *((_QWORD *)v10 + 2);
        v13[1] = 0;
        v13[2] = v14;
        if ( v14 != -4096 && v14 != 0 && v14 != -8192 )
          sub_BD6050(v13, *(_QWORD *)v10 & 0xFFFFFFFFFFFFFFF8LL);
        LODWORD(v4) = *(_DWORD *)(a1 + 224);
      }
      *(_DWORD *)(a1 + 224) = v4 + 1;
      LOBYTE(v4) = (_BYTE)v19;
      if ( v19 + 4096 != 0 && v19 != 0 && v19 != (unsigned __int8 *)-8192LL )
        LOBYTE(v4) = sub_BD60C0(v18);
    }
  }
  return v4;
}
