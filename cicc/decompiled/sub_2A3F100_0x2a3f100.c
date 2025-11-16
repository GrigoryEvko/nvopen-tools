// Function: sub_2A3F100
// Address: 0x2a3f100
//
__int64 __fastcall sub_2A3F100(_QWORD **a1, __int64 a2, unsigned __int64 a3, const void *a4, __int64 a5, char a6)
{
  __int64 *v9; // rax
  unsigned __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // r13
  __int64 v15; // [rsp+8h] [rbp-38h]

  v9 = (__int64 *)sub_BCB120(*a1);
  v10 = sub_BCF480(v9, a4, a5, 0);
  v12 = sub_BA8CA0((__int64)a1, a2, a3, v10);
  if ( a6 )
  {
    v15 = v11;
    if ( sub_B2FC80(v11) )
      *(_BYTE *)(v15 + 32) = *(_BYTE *)(v15 + 32) & 0xF0 | 9;
  }
  return v12;
}
