// Function: sub_16C2D20
// Address: 0x16c2d20
//
__int64 __fastcall sub_16C2D20(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        unsigned __int64 a4,
        __int64 a5,
        char a6,
        unsigned __int8 a7)
{
  int v9; // eax
  __int64 v10; // rdx
  __int64 v12; // rax
  int v13; // eax
  int fd; // [rsp+1Ch] [rbp-54h] BYREF
  _QWORD v17[2]; // [rsp+20h] [rbp-50h] BYREF
  char v18; // [rsp+30h] [rbp-40h]

  v9 = sub_16C5BD0(a2, &fd, 0, 0);
  if ( v9 )
  {
    *(_BYTE *)(a1 + 16) |= 1u;
    *(_DWORD *)a1 = v9;
    *(_QWORD *)(a1 + 8) = v10;
  }
  else
  {
    sub_16C2910((unsigned __int64)v17, (__int128 *)(unsigned int)fd, a2, a3, a4, a5, a6, a7);
    close(fd);
    if ( (v18 & 1) != 0 )
    {
      v13 = v17[0];
      *(_BYTE *)(a1 + 16) |= 1u;
      *(_DWORD *)a1 = v13;
      *(_QWORD *)(a1 + 8) = v17[1];
    }
    else
    {
      v12 = v17[0];
      *(_BYTE *)(a1 + 16) &= ~1u;
      *(_QWORD *)a1 = v12;
    }
  }
  return a1;
}
