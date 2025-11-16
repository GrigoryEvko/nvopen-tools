// Function: sub_3108E30
// Address: 0x3108e30
//
__int64 __fastcall sub_3108E30(__int64 a1, __int64 a2)
{
  unsigned int v2; // r12d
  char *v4; // [rsp+0h] [rbp-40h] BYREF
  unsigned int v5; // [rsp+8h] [rbp-38h]
  unsigned __int64 v6; // [rsp+10h] [rbp-30h] BYREF
  unsigned int v7; // [rsp+18h] [rbp-28h]

  v2 = *(unsigned __int8 *)(a1 + 344);
  if ( !(_BYTE)v2 )
    return v2;
  v5 = sub_AE43F0(*(_QWORD *)a1, *(_QWORD *)(a2 + 8));
  if ( v5 > 0x40 )
    sub_C43690((__int64)&v4, 0, 0);
  else
    v4 = 0;
  v2 = sub_B4DE60(a2, *(_QWORD *)a1, (__int64)&v4);
  if ( (_BYTE)v2 )
  {
    sub_C44B10((__int64)&v6, &v4, *(_DWORD *)(a1 + 360));
    sub_C45EE0(a1 + 352, (__int64 *)&v6);
    if ( v7 > 0x40 )
    {
      if ( v6 )
        j_j___libc_free_0_0(v6);
    }
  }
  if ( v5 <= 0x40 || !v4 )
    return v2;
  j_j___libc_free_0_0((unsigned __int64)v4);
  return v2;
}
