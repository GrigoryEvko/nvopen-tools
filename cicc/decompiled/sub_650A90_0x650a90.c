// Function: sub_650A90
// Address: 0x650a90
//
void __fastcall sub_650A90(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // [rsp-20h] [rbp-20h]

  if ( unk_4D0418C )
  {
    v1 = sub_5CC190(23);
    if ( v1 )
    {
      if ( (_DWORD)qword_4F077B4 && *(_BYTE *)(v1 + 9) == 1 )
      {
        v2 = v1;
        sub_684B30(2408, v1 + 56);
        v1 = v2;
      }
      if ( a1 )
        sub_5CEC90((_QWORD *)v1, a1, 29);
    }
  }
}
