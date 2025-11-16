// Function: sub_3735CB0
// Address: 0x3735cb0
//
__int64 __fastcall sub_3735CB0(__int64 a1)
{
  __int64 result; // rax
  __int64 v2; // rbx
  __int64 v3; // rax
  __int64 v4; // rcx

  result = *(_QWORD *)(a1 + 80);
  if ( *(_DWORD *)(result + 32) != 3 )
  {
    v2 = sub_31DA6B0(*(_QWORD *)(a1 + 184));
    if ( *(_BYTE *)(*(_QWORD *)(a1 + 208) + 3689LL) )
    {
      v4 = *(_QWORD *)(*(_QWORD *)(v2 + 96) + 16LL);
      *(_QWORD *)(a1 + 400) = v4;
    }
    else
    {
      v3 = (*(__int64 (__fastcall **)(_QWORD, _QWORD))(**(_QWORD **)(*(_QWORD *)(a1 + 184) + 224LL) + 848LL))(
             *(_QWORD *)(*(_QWORD *)(a1 + 184) + 224LL),
             *(unsigned int *)(a1 + 72));
      *(_QWORD *)(a1 + 400) = v3;
      v4 = v3;
    }
    return sub_324AC60((__int64 *)a1, a1 + 8, 16, v4, *(_QWORD *)(*(_QWORD *)(v2 + 96) + 16LL));
  }
  return result;
}
