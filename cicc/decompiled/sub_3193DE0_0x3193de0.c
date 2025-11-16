// Function: sub_3193DE0
// Address: 0x3193de0
//
__int64 __fastcall sub_3193DE0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rax
  unsigned __int64 v6; // rdi
  int v8; // eax
  __int64 v9; // r14
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // r15
  unsigned __int16 v18; // r13
  _QWORD *v19; // rdi
  __int64 v20; // [rsp+0h] [rbp-30h] BYREF
  unsigned __int16 v21; // [rsp+8h] [rbp-28h]

  v5 = a1 + 48;
  v6 = *(_QWORD *)(a1 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v6 == v5 )
    goto LABEL_36;
  if ( !v6 )
    BUG();
  if ( (unsigned int)*(unsigned __int8 *)(v6 - 24) - 30 > 0xA )
LABEL_36:
    BUG();
  v8 = *(_DWORD *)(v6 - 20) & 0x7FFFFFF;
  if ( v8 == 3 )
  {
    v9 = *(_QWORD *)(v6 - 120);
  }
  else
  {
    v9 = 0;
    if ( v8 == 1 )
    {
      if ( *(_QWORD *)(v6 - 56) )
      {
        v13 = *(_QWORD *)(v6 - 48);
        **(_QWORD **)(v6 - 40) = v13;
        if ( v13 )
          *(_QWORD *)(v13 + 16) = *(_QWORD *)(v6 - 40);
      }
      *(_QWORD *)(v6 - 56) = a4;
      v9 = 0;
      if ( a4 )
      {
        v14 = *(_QWORD *)(a4 + 16);
        *(_QWORD *)(v6 - 48) = v14;
        if ( v14 )
          *(_QWORD *)(v14 + 16) = v6 - 48;
        *(_QWORD *)(v6 - 40) = a4 + 16;
        v9 = 0;
        *(_QWORD *)(a4 + 16) = v6 - 56;
      }
      return v9;
    }
  }
  if ( !a2 )
  {
    if ( !a3 )
    {
LABEL_33:
      sub_B43D60((_QWORD *)(v6 - 24));
      sub_B43C20((__int64)&v20, a1);
      v17 = v20;
      v18 = v21;
      v19 = sub_BD2C40(72, 1u);
      if ( v19 )
        sub_B4C8F0((__int64)v19, a4, 1u, v17, v18);
      return v9;
    }
LABEL_8:
    if ( !a2 )
    {
      if ( *(_QWORD *)(v6 - 88) )
      {
        v10 = *(_QWORD *)(v6 - 80);
        **(_QWORD **)(v6 - 72) = v10;
        if ( v10 )
          *(_QWORD *)(v10 + 16) = *(_QWORD *)(v6 - 72);
      }
      *(_QWORD *)(v6 - 88) = a4;
      if ( a4 )
      {
        v11 = *(_QWORD *)(a4 + 16);
        *(_QWORD *)(v6 - 80) = v11;
        if ( v11 )
          *(_QWORD *)(v11 + 16) = v6 - 80;
        *(_QWORD *)(v6 - 72) = a4 + 16;
        *(_QWORD *)(a4 + 16) = v6 - 88;
      }
      return v9;
    }
    goto LABEL_33;
  }
  if ( a3 )
    goto LABEL_8;
  if ( *(_QWORD *)(v6 - 56) )
  {
    v15 = *(_QWORD *)(v6 - 48);
    **(_QWORD **)(v6 - 40) = v15;
    if ( v15 )
      *(_QWORD *)(v15 + 16) = *(_QWORD *)(v6 - 40);
  }
  *(_QWORD *)(v6 - 56) = a4;
  if ( a4 )
  {
    v16 = *(_QWORD *)(a4 + 16);
    *(_QWORD *)(v6 - 48) = v16;
    if ( v16 )
      *(_QWORD *)(v16 + 16) = v6 - 48;
    *(_QWORD *)(v6 - 40) = a4 + 16;
    *(_QWORD *)(a4 + 16) = v6 - 56;
  }
  return v9;
}
