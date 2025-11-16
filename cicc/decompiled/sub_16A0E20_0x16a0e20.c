// Function: sub_16A0E20
// Address: 0x16a0e20
//
__int64 __fastcall sub_16A0E20(__int64 a1, __int64 a2, unsigned int a3, double a4, double a5, double a6)
{
  void *v8; // rax
  void *v9; // rbx
  _BYTE *v11; // rdi
  char *v12; // rax
  char v13; // al

  v8 = sub_16982C0();
  if ( v8 == *(void **)(a1 + 8) )
    return sub_16A0E00((__int64 *)(a1 + 8), (__int64 *)(a2 + 8), a3, a4, a5, a6);
  v9 = v8;
  if ( (unsigned __int8)sub_169DE70(a1) || (unsigned __int8)sub_169DE70(a2) )
  {
    v11 = (_BYTE *)(a1 + 8);
    if ( v9 == *(void **)(a1 + 8) )
      sub_169CAA0((__int64)v11, 0, 0, 0, *(float *)&a4);
    else
      sub_16986F0(v11, 0, 0, 0);
    return 1;
  }
  else if ( *(void **)(a1 + 8) == sub_1698270()
         && ((v12 = (char *)sub_16D40F0(qword_4FBB490)) == 0 ? (v13 = qword_4FBB490[2]) : (v13 = *v12), v13) )
  {
    return sub_169FD40(a1, a2, a3);
  }
  else
  {
    return sub_169CEB0((__int16 **)(a1 + 8), (_BYTE *)(a2 + 8), a3);
  }
}
