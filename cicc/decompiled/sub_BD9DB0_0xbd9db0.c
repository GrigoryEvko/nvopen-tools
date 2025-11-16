// Function: sub_BD9DB0
// Address: 0xbd9db0
//
__int64 __fastcall sub_BD9DB0(__int64 *a1)
{
  __int64 v1; // rcx
  unsigned __int64 v2; // r8
  __int64 v3; // r9
  __int64 result; // rax
  __int64 v5; // rdx
  __int64 v6; // r8
  unsigned __int8 v7; // al

  v1 = *a1;
  while ( 1 )
  {
    v5 = *(a1 - 1);
    v6 = *(_QWORD *)(*(_QWORD *)(v1 - 32LL * (*(_DWORD *)(v1 + 4) & 0x7FFFFFF)) + 24LL);
    v7 = *(_BYTE *)(v6 - 16);
    v2 = (v7 & 2) != 0 ? *(_QWORD *)(v6 - 32) : -16 - 8LL * ((v7 >> 2) & 0xF) + v6;
    v3 = *(_QWORD *)(*(_QWORD *)(v5 - 32LL * (*(_DWORD *)(v5 + 4) & 0x7FFFFFF)) + 24LL);
    result = *(unsigned __int8 *)(v3 - 16);
    if ( (result & 2) == 0 )
      break;
    if ( *(_QWORD *)(v3 - 32) <= v2 )
      goto LABEL_9;
LABEL_5:
    *a1-- = v5;
  }
  result = 8LL * (((unsigned __int8)result >> 2) & 0xF);
  if ( -16 - result + v3 > v2 )
    goto LABEL_5;
LABEL_9:
  *a1 = v1;
  return result;
}
