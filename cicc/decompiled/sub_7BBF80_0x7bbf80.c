// Function: sub_7BBF80
// Address: 0x7bbf80
//
__int64 __fastcall sub_7BBF80(unsigned __int64 a1, unsigned int *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rdx
  __int64 v7; // rax
  __int64 result; // rax

  if ( *(_QWORD *)(a1 + 8) )
  {
    if ( !(_DWORD)a2 )
      sub_7AE360(a1);
    v6 = qword_4F08560;
    **(_QWORD **)(a1 + 16) = qword_4F08560;
    v7 = *(_QWORD *)(a1 + 8);
    *(_QWORD *)(a1 + 16) = 0;
    qword_4F08560 = v7;
    *(_QWORD *)(a1 + 8) = 0;
    dword_4F061FC = 1;
    return sub_7B8B50(a1, a2, v6, a4, a5, a6);
  }
  else if ( (_DWORD)a2 )
  {
    return sub_7B8B50(a1, a2, a3, a4, a5, a6);
  }
  return result;
}
