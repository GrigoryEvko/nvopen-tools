// Function: sub_37C7360
// Address: 0x37c7360
//
unsigned __int64 __fastcall sub_37C7360(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  int v5; // eax
  __int64 v6; // rdi
  unsigned __int64 v7; // rbx
  char v9; // dl
  char v10; // dl
  __int64 v11; // [rsp+0h] [rbp-40h]
  unsigned __int64 v12; // [rsp+10h] [rbp-30h]

  v2 = *(_QWORD *)(a2 + 48);
  v3 = v2 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (v2 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
    goto LABEL_12;
  if ( (v2 & 7) != 0 )
  {
    if ( (v2 & 7) == 3 && *(_DWORD *)v3 == 1 )
      goto LABEL_4;
LABEL_12:
    BYTE4(v11) = 0;
    return v11;
  }
  *(_QWORD *)(a2 + 48) = v3;
  LOBYTE(v2) = v2 & 0xF8;
LABEL_4:
  v5 = v2 & 7;
  if ( v5 )
  {
    if ( v5 != 3 )
      BUG();
    v3 = *(_QWORD *)(v3 + 16);
  }
  else
  {
    *(_QWORD *)(a2 + 48) = v3;
  }
  v6 = *(_QWORD *)v3;
  if ( !*(_QWORD *)v3 || (v6 & 4) == 0 )
    BUG();
  LOBYTE(v7) = 0;
  if ( !(*(unsigned __int8 (__fastcall **)(unsigned __int64, _QWORD))(*(_QWORD *)(v6 & 0xFFFFFFFFFFFFFFF8LL) + 32LL))(
          v6 & 0xFFFFFFFFFFFFFFF8LL,
          *(_QWORD *)(a1 + 48)) )
  {
    sub_2E8E2F0(a2, *(_QWORD *)(a1 + 32));
    if ( v9 || (v12 = sub_2E8E430(a2, *(__int64 **)(a1 + 32)), v10) )
    {
      v12 = sub_37C70E0(a1, a2);
      v7 = HIDWORD(v12);
    }
  }
  BYTE4(v12) = v7;
  return v12;
}
