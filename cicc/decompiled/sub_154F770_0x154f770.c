// Function: sub_154F770
// Address: 0x154f770
//
_BYTE *__fastcall sub_154F770(__int64 a1, unsigned __int8 *a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r13
  int v6; // eax
  _BYTE *result; // rax
  __int64 v9; // rdx
  __int64 v10; // r14
  int v11; // ebx
  _BYTE *v12; // rax
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // r8
  __int64 v20; // [rsp+8h] [rbp-28h]
  __int64 v22; // [rsp+8h] [rbp-28h]

  v5 = a1;
  v6 = *a2;
  if ( (_BYTE)v6 == 6 )
    return sub_15499D0(a1, (__int64)a2);
  v9 = (unsigned int)(v6 - 4);
  if ( (unsigned __int8)(v6 - 4) > 0x1Eu )
  {
    if ( (_BYTE)v6 )
    {
      v20 = a4;
      sub_154DAA0(a3, **((_QWORD **)a2 + 17), a1);
      sub_1549FC0(a1, 0x20u);
      return (_BYTE *)sub_1550E20(a1, *((_QWORD *)a2 + 17), a3, v20, a5);
    }
    else
    {
      sub_1263B40(a1, "!\"");
      v13 = sub_161E970(a2);
      sub_16D16F0(v13, v14, a1);
      return (_BYTE *)sub_1549FC0(a1, 0x22u);
    }
  }
  else
  {
    v10 = 0;
    if ( !a4 )
    {
      v17 = sub_22077B0(272);
      a4 = v17;
      if ( v17 )
      {
        v18 = a5;
        v22 = v17;
        sub_154BB30(v17, v18, 0);
        a4 = v22;
      }
      v10 = a4;
    }
    v11 = sub_154F490(a4, (__int64)a2, v9, a4);
    if ( v11 == -1 )
    {
      v15 = sub_1263B40(a1, "<");
      v16 = sub_16E7B40(v15, a2);
      result = (_BYTE *)sub_1263B40(v16, ">");
    }
    else
    {
      v12 = *(_BYTE **)(a1 + 24);
      if ( (unsigned __int64)v12 >= *(_QWORD *)(a1 + 16) )
      {
        v5 = sub_16E7DE0(a1, 33);
      }
      else
      {
        *(_QWORD *)(a1 + 24) = v12 + 1;
        *v12 = 33;
      }
      result = (_BYTE *)sub_16E7AB0(v5, v11);
    }
    if ( v10 )
    {
      sub_154C1A0(v10);
      return (_BYTE *)j_j___libc_free_0(v10, 272);
    }
  }
  return result;
}
