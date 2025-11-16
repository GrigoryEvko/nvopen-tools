// Function: sub_1806600
// Address: 0x1806600
//
void __fastcall sub_1806600(
        _QWORD *a1,
        __int64 a2,
        __int64 a3,
        unsigned __int64 a4,
        unsigned __int64 a5,
        __int64 *a6,
        __int64 a7)
{
  unsigned __int64 v10; // rdx
  unsigned __int64 v11; // rdi
  _QWORD *v12; // r10
  unsigned __int64 v13; // r8
  __int64 v14; // r15
  char v15; // al
  unsigned __int64 v16; // rbx
  __int64 v17; // rdi
  __int64 v18; // rax
  __int64 v19; // [rsp+18h] [rbp-A8h]
  __int64 v20; // [rsp+20h] [rbp-A0h]
  _QWORD *v21; // [rsp+28h] [rbp-98h]
  _QWORD v23[2]; // [rsp+40h] [rbp-80h] BYREF
  _BYTE v24[16]; // [rsp+50h] [rbp-70h] BYREF
  __int16 v25; // [rsp+60h] [rbp-60h]
  _BYTE v26[16]; // [rsp+70h] [rbp-50h] BYREF
  __int16 v27; // [rsp+80h] [rbp-40h]

  if ( a4 >= a5 )
  {
    v11 = a4;
  }
  else
  {
    v10 = a4 + 1;
    v11 = a4;
    v12 = a1;
    v13 = a4;
    do
    {
      if ( *(_BYTE *)(a2 + v13) && (v14 = *(unsigned __int8 *)(a3 + v13), v12[v14 + 135]) )
      {
        while ( a5 > v10 )
        {
          v15 = *(_BYTE *)(a2 + v10);
          v16 = v10++;
          if ( !v15 || *(_BYTE *)(a3 + v13) != *(_BYTE *)(a3 + v10 - 1) )
            goto LABEL_10;
        }
        v16 = v10;
LABEL_10:
        if ( v16 - v13 >= (unsigned int)dword_4FA84A0 )
        {
          v19 = v16 - v13;
          v20 = v13;
          v21 = v12;
          sub_1804D00(v12, a2, a3, v11, v13, (__int64)a6, a7);
          v25 = 257;
          v17 = v21[61];
          v27 = 257;
          v18 = sub_15A0680(v17, v20, 0);
          v23[0] = sub_12899C0(a6, a7, v18, (__int64)v24, 0, 0);
          v23[1] = sub_15A0680(v21[61], v19, 0);
          sub_1285290(a6, *(_QWORD *)(v21[v14 + 135] + 24LL), v21[v14 + 135], (int)v23, 2, (__int64)v26, 0);
          v12 = v21;
          v13 = v16;
          v11 = v16;
        }
        else
        {
          v13 = v16;
        }
      }
      else
      {
        v13 = v10;
      }
      v10 = v13 + 1;
    }
    while ( a5 > v13 );
    a1 = v12;
  }
  sub_1804D00(a1, a2, a3, v11, a5, (__int64)a6, a7);
}
