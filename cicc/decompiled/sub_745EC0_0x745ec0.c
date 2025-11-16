// Function: sub_745EC0
// Address: 0x745ec0
//
__int64 __fastcall sub_745EC0(__int64 a1, _DWORD *a2, __int64 a3)
{
  __int64 i; // rax
  __int64 **v6; // r14
  char v7; // al
  __int64 result; // rax
  __int64 *v9; // rbx
  unsigned int v10; // r12d
  unsigned __int64 v11; // rsi

  for ( i = a1; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  v6 = *(__int64 ***)(i + 168);
  sub_745E10(a1, a2, (__int64 (__fastcall **)(const char *, _QWORD))a3);
  v7 = *((_BYTE *)v6 + 20);
  if ( (v7 & 4) == 0 || *(_BYTE *)(a3 + 136) && *(_BYTE *)(a3 + 141) )
  {
    if ( (v7 & 1) == 0 )
      goto LABEL_7;
LABEL_24:
    sub_7450F0("__noreturn__", a2, (__int64 (__fastcall **)(const char *, _QWORD))a3);
    if ( (*((_BYTE *)v6 + 20) & 8) == 0 )
      goto LABEL_8;
    goto LABEL_25;
  }
  sub_7450F0("__warn_unused_result__", a2, (__int64 (__fastcall **)(const char *, _QWORD))a3);
  v7 = *((_BYTE *)v6 + 20);
  if ( (v7 & 1) != 0 )
    goto LABEL_24;
LABEL_7:
  if ( (v7 & 8) == 0 )
    goto LABEL_8;
LABEL_25:
  sub_7450F0("__const__", a2, (__int64 (__fastcall **)(const char *, _QWORD))a3);
LABEL_8:
  result = unk_4F06A80 | (unsigned int)(unk_4F06A7C | unk_4F06A78);
  if ( !(unk_4F06A80 | unk_4F06A7C | unk_4F06A78) )
  {
    result = *((unsigned __int8 *)v6 + 25);
    switch ( *((_BYTE *)v6 + 25) )
    {
      case 0:
      case 4:
      case 5:
      case 6:
        break;
      case 1:
        result = sub_7450F0("__cdecl__", a2, (__int64 (__fastcall **)(const char *, _QWORD))a3);
        break;
      case 2:
        if ( *(_BYTE *)(a3 + 136) )
        {
          result = (__int64)&qword_4F068D8;
          if ( qword_4F068D8 <= 0x9D07u )
            goto LABEL_21;
        }
        result = sub_7450F0("__fastcall__", a2, (__int64 (__fastcall **)(const char *, _QWORD))a3);
        break;
      case 3:
        result = sub_7450F0("__stdcall__", a2, (__int64 (__fastcall **)(const char *, _QWORD))a3);
        if ( *(_BYTE *)(a3 + 136) )
          goto LABEL_21;
        goto LABEL_12;
      default:
        sub_721090();
    }
  }
  if ( *(_BYTE *)(a3 + 136) )
  {
LABEL_21:
    if ( *(_BYTE *)(a3 + 141) )
      return result;
  }
LABEL_12:
  v9 = *v6;
  if ( *v6 )
  {
    v10 = 1;
    do
    {
      while ( (*((_BYTE *)v9 + 34) & 8) == 0 )
      {
        v9 = (__int64 *)*v9;
        ++v10;
        if ( !v9 )
          goto LABEL_17;
      }
      v11 = v10++;
      sub_745D60("nonnull", v11, a2, (__int64 (__fastcall **)(const char *, _QWORD))a3);
      v9 = (__int64 *)*v9;
    }
    while ( v9 );
  }
LABEL_17:
  result = *((unsigned int *)v6 + 9);
  if ( (_DWORD)result )
    return sub_745D60("sentinel", (int)result - 1, a2, (__int64 (__fastcall **)(const char *, _QWORD))a3);
  return result;
}
