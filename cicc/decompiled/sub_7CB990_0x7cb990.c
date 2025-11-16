// Function: sub_7CB990
// Address: 0x7cb990
//
__int64 __fastcall sub_7CB990(__int64 a1, unsigned __int64 a2, int a3)
{
  int v3; // ebx
  __int64 result; // rax

  if ( a3 | HIDWORD(qword_4F077B4) | (unk_4F064A8 != 0) )
  {
    v3 = sub_722A20(a2, (_BYTE *)(a1 + 45)) - 1;
    result = *(char *)(a1 + 45);
  }
  else
  {
    v3 = 0;
    *(_BYTE *)(a1 + 45) = a2;
    result = (char)a2;
    if ( a2 > 0xFF )
    {
      sub_7B0EB0(**(_QWORD **)a1, (__int64)dword_4F07508);
      sub_684B30(0x8EAu, dword_4F07508);
      result = *(char *)(a1 + 45);
    }
  }
  *(_DWORD *)(a1 + 16) = v3;
  *(_QWORD *)(a1 + 24) = a1 + 46;
  return result;
}
