// Function: sub_38ED5E0
// Address: 0x38ed5e0
//
__int64 __fastcall sub_38ED5E0(unsigned __int8 **a1)
{
  __int64 v2; // rdi
  bool v3; // zf
  unsigned int v4; // r12d
  _QWORD *v6; // [rsp+0h] [rbp-40h] BYREF
  __int64 v7; // [rsp+8h] [rbp-38h]
  _BYTE v8[48]; // [rsp+10h] [rbp-30h] BYREF

  v2 = (__int64)*a1;
  v6 = v8;
  v7 = 0;
  v3 = *(_BYTE *)(v2 + 845) == 0;
  v8[0] = 0;
  if ( v3 )
  {
    if ( (unsigned __int8)sub_38E36C0(v2) )
    {
LABEL_10:
      v4 = 1;
      goto LABEL_7;
    }
    v2 = (__int64)*a1;
  }
  if ( (unsigned __int8)sub_38ECF20(v2, (unsigned __int64 *)&v6) )
    goto LABEL_10;
  (*(void (__fastcall **)(_QWORD, _QWORD *, __int64))(**((_QWORD **)*a1 + 41) + 400LL))(*((_QWORD *)*a1 + 41), v6, v7);
  v4 = *a1[1];
  if ( (_BYTE)v4 )
  {
    v4 = 0;
    (*(void (__fastcall **)(_QWORD, void *, __int64))(**((_QWORD **)*a1 + 41) + 400LL))(
      *((_QWORD *)*a1 + 41),
      &unk_452ECA4,
      1);
  }
LABEL_7:
  if ( v6 != (_QWORD *)v8 )
    j_j___libc_free_0((unsigned __int64)v6);
  return v4;
}
