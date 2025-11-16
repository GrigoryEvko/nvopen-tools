// Function: sub_FFF250
// Address: 0xfff250
//
__int64 __fastcall sub_FFF250(__int64 a1, __int64 a2, unsigned __int8 **a3)
{
  unsigned __int8 *v4; // rax
  unsigned int v5; // eax
  char *v7; // [rsp+0h] [rbp-30h] BYREF
  unsigned int v8; // [rsp+8h] [rbp-28h]

  v8 = sub_AE43F0(a2, *((_QWORD *)*a3 + 1));
  if ( v8 > 0x40 )
    sub_C43690((__int64)&v7, 0, 0);
  else
    v7 = 0;
  v4 = sub_BD45C0(*a3, a2, (__int64)&v7, 0, 0, 0, 0, 0);
  *a3 = v4;
  v5 = sub_AE43F0(a2, *((_QWORD *)v4 + 1));
  sub_C44B10(a1, &v7, v5);
  if ( v8 > 0x40 && v7 )
    j_j___libc_free_0_0(v7);
  return a1;
}
