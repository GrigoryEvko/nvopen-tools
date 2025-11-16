// Function: sub_392BF20
// Address: 0x392bf20
//
bool __fastcall sub_392BF20(__int64 a1, const char *a2)
{
  __int64 v4; // rax
  size_t v5; // rdx
  __int64 v6; // rsi

  v4 = *(_QWORD *)(a1 + 136);
  v5 = *(_QWORD *)(v4 + 56);
  v6 = *(_QWORD *)(v4 + 48);
  if ( v5 == 1 || *(_BYTE *)(v6 + 1) == 35 )
    return *a2 == *(_BYTE *)v6;
  else
    return strncmp(a2, (const char *)v6, v5) == 0;
}
