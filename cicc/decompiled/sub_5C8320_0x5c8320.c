// Function: sub_5C8320
// Address: 0x5c8320
//
__int64 __fastcall sub_5C8320(__int64 a1, int a2)
{
  __int64 v2; // r12
  __int64 v3; // rax
  __int64 v4; // rdi
  _DWORD v6[5]; // [rsp+Ch] [rbp-14h] BYREF

  if ( unk_4F06418 == 7 )
  {
    if ( unk_4F063AD )
    {
      if ( !a2 || (unsigned int)sub_72AE80(&unk_4F06300) )
      {
        v2 = sub_7276D0();
        *(_BYTE *)(v2 + 10) = 3;
        *(_QWORD *)(v2 + 24) = unk_4F063F8;
        *(_QWORD *)(v2 + 32) = unk_4F063F0;
        sub_7296C0(v6);
        v3 = sub_73A460(&unk_4F06300);
        v4 = v6[0];
        *(_QWORD *)(v2 + 40) = v3;
        sub_729730(v4);
        sub_7B8B50();
        return v2;
      }
      sub_6851C0(1434, &unk_4F063F8);
      sub_7B8B50();
    }
    else
    {
      sub_7B8B50();
    }
  }
  else
  {
    sub_6851D0(1038);
  }
  *(_BYTE *)(a1 + 8) = 0;
  return 0;
}
