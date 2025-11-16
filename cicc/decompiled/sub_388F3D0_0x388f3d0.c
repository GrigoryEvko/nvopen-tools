// Function: sub_388F3D0
// Address: 0x388f3d0
//
__int64 __fastcall sub_388F3D0(__int64 a1, _BYTE *a2)
{
  __int64 v3; // rdi
  unsigned int v4; // eax
  unsigned __int64 v6; // rsi
  const char *v7; // [rsp+0h] [rbp-30h] BYREF
  char v8; // [rsp+10h] [rbp-20h]
  char v9; // [rsp+11h] [rbp-1Fh]

  v3 = a1 + 8;
  v4 = *(_DWORD *)(v3 + 56);
  if ( v4 == 324 )
  {
    *a2 = 0;
    goto LABEL_6;
  }
  if ( v4 <= 0x144 )
  {
    if ( v4 == 52 )
    {
      *a2 = 2;
      goto LABEL_6;
    }
    if ( v4 == 151 )
    {
      *a2 = 1;
      goto LABEL_6;
    }
  }
  else
  {
    if ( v4 == 325 )
    {
      *a2 = 3;
      goto LABEL_6;
    }
    if ( v4 == 326 )
    {
      *a2 = 4;
LABEL_6:
      *(_DWORD *)(a1 + 64) = sub_3887100(v3);
      return 0;
    }
  }
  v6 = *(_QWORD *)(a1 + 56);
  v9 = 1;
  v7 = "invalid call edge hotness";
  v8 = 3;
  return sub_38814C0(v3, v6, (__int64)&v7);
}
