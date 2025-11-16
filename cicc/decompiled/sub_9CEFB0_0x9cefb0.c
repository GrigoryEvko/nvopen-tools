// Function: sub_9CEFB0
// Address: 0x9cefb0
//
__int64 __fastcall sub_9CEFB0(__int64 a1, __int64 a2, char a3, __int64 a4)
{
  __int64 v5; // rdx
  char v6; // al
  char v7; // al
  __int64 v9; // rax
  __int64 v10; // [rsp+8h] [rbp-48h] BYREF
  __int64 v11; // [rsp+10h] [rbp-40h] BYREF
  char v12; // [rsp+18h] [rbp-38h]

  while ( 1 )
  {
    sub_9CEA50((__int64)&v11, a2, a3, a4);
    v5 = v12 & 1;
    v6 = (2 * v5) | v12 & 0xFD;
    v12 = v6;
    if ( (_BYTE)v5 )
    {
      *(_BYTE *)(a1 + 8) |= 3u;
      v12 = v6 & 0xFD;
      v9 = v11;
      v11 = 0;
      *(_QWORD *)a1 = v9;
LABEL_13:
      if ( v11 )
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v11 + 8LL))(v11);
      return a1;
    }
    if ( (_DWORD)v11 != 2 )
    {
      *(_BYTE *)(a1 + 8) = *(_BYTE *)(a1 + 8) & 0xFC | 2;
      *(_QWORD *)a1 = v11;
      return a1;
    }
    sub_9CE5C0(&v10, a2, v5, (unsigned int)(2 * v5));
    if ( (v10 & 0xFFFFFFFFFFFFFFFELL) != 0 )
      break;
    if ( (v12 & 2) != 0 )
      goto LABEL_11;
    if ( (v12 & 1) != 0 && v11 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v11 + 8LL))(v11);
  }
  *(_QWORD *)a1 = v10 & 0xFFFFFFFFFFFFFFFELL;
  v7 = v12;
  *(_BYTE *)(a1 + 8) |= 3u;
  if ( (v7 & 2) != 0 )
LABEL_11:
    sub_9CEF10(&v11);
  if ( (v7 & 1) != 0 )
    goto LABEL_13;
  return a1;
}
