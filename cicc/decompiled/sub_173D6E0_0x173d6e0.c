// Function: sub_173D6E0
// Address: 0x173d6e0
//
__int64 __fastcall sub_173D6E0(__int64 a1, __int64 a2, unsigned int a3, float a4)
{
  void *v6; // rax
  void *v7; // rbx
  _BYTE *v9; // rdi
  char *v10; // rax
  char v11; // al

  v6 = sub_16982C0();
  if ( v6 == *(void **)(a1 + 8) )
    return sub_169EEB0((_QWORD *)(a1 + 8), a2 + 8, a3, a4);
  v7 = v6;
  if ( (unsigned __int8)sub_169DE70(a1) || (unsigned __int8)sub_169DE70(a2) )
  {
    v9 = (_BYTE *)(a1 + 8);
    if ( v7 == *(void **)(a1 + 8) )
      sub_169CAA0((__int64)v9, 0, 0, 0, a4);
    else
      sub_16986F0(v9, 0, 0, 0);
    return 1;
  }
  else if ( *(void **)(a1 + 8) == sub_1698270()
         && ((v10 = (char *)sub_16D40F0((__int64)qword_4FBB490)) == 0 ? (v11 = qword_4FBB490[2]) : (v11 = *v10), v11) )
  {
    return sub_1581A10(a1, a2, a3, a4);
  }
  else
  {
    return sub_16994B0((__int16 **)(a1 + 8), a2 + 8, a3);
  }
}
