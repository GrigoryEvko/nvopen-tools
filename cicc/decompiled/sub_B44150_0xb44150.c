// Function: sub_B44150
// Address: 0xb44150
//
void __fastcall sub_B44150(_QWORD *a1, __int64 a2, unsigned __int64 *a3, __int64 a4)
{
  unsigned __int64 v6; // rcx
  __int64 v7; // rax
  __int64 v8; // rax

  sub_AA48C0(a2 + 48, (__int64)a1);
  v6 = *a3;
  v7 = a1[3];
  a1[4] = a3;
  v6 &= 0xFFFFFFFFFFFFFFF8LL;
  a1[3] = v6 | v7 & 7;
  *(_QWORD *)(v6 + 8) = a1 + 3;
  *a3 = *a3 & 7 | (unsigned __int64)(a1 + 3);
  if ( *(_BYTE *)(a2 + 40) )
  {
    if ( !(_BYTE)a4 )
    {
      v8 = sub_AA6160(a2, (__int64)a3);
      if ( v8 )
      {
        if ( v8 + 8 != (*(_QWORD *)(v8 + 8) & 0xFFFFFFFFFFFFFFF8LL) )
          sub_B44050((__int64)a1, a2, (__int64)a3, a4, 0);
      }
    }
    if ( (unsigned int)*(unsigned __int8 *)a1 - 30 <= 0xA )
      sub_AA6320(a1[5]);
  }
}
