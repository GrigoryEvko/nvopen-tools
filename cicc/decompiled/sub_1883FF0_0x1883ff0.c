// Function: sub_1883FF0
// Address: 0x1883ff0
//
__int64 __fastcall sub_1883FF0(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  char v3; // [rsp+7h] [rbp-29h] BYREF
  _QWORD v4[5]; // [rsp+8h] [rbp-28h] BYREF

  if ( (*(unsigned __int8 (__fastcall **)(__int64, char *, _QWORD, _QWORD, char *, _QWORD *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "Linkage",
         0,
         0,
         &v3,
         v4) )
  {
    sub_1879C60(a1, (unsigned int *)a2);
    (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, v4[0]);
  }
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _QWORD, char *, _QWORD *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "NotEligibleToImport",
         0,
         0,
         &v3,
         v4) )
  {
    sub_1879DE0(a1, (_BYTE *)(a2 + 4));
    (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, v4[0]);
  }
  if ( (*(unsigned __int8 (__fastcall **)(__int64, char *, _QWORD, _QWORD, char *, _QWORD *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "Live",
         0,
         0,
         &v3,
         v4) )
  {
    sub_1879DE0(a1, (_BYTE *)(a2 + 5));
    (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, v4[0]);
  }
  if ( (*(unsigned __int8 (__fastcall **)(__int64, char *, _QWORD, _QWORD, char *, _QWORD *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "Local",
         0,
         0,
         &v3,
         v4) )
  {
    sub_1879DE0(a1, (_BYTE *)(a2 + 6));
    (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, v4[0]);
  }
  if ( (!(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 56LL))(a1)
     || *(_QWORD *)(a2 + 16) != *(_QWORD *)(a2 + 8))
    && (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _QWORD, char *, _QWORD *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "Refs",
         0,
         0,
         &v3,
         v4) )
  {
    sub_1883700(a1, (__int64 *)(a2 + 8));
    (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, v4[0]);
  }
  if ( (!(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 56LL))(a1)
     || *(_QWORD *)(a2 + 40) != *(_QWORD *)(a2 + 32))
    && (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _QWORD, char *, _QWORD *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "TypeTests",
         0,
         0,
         &v3,
         v4) )
  {
    sub_1883700(a1, (__int64 *)(a2 + 32));
    (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, v4[0]);
  }
  if ( (!(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 56LL))(a1)
     || *(_QWORD *)(a2 + 64) != *(_QWORD *)(a2 + 56))
    && (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _QWORD, char *, _QWORD *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "TypeTestAssumeVCalls",
         0,
         0,
         &v3,
         v4) )
  {
    sub_18839E0(a1, (__int64 *)(a2 + 56));
    (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, v4[0]);
  }
  if ( (!(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 56LL))(a1)
     || *(_QWORD *)(a2 + 88) != *(_QWORD *)(a2 + 80))
    && (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _QWORD, char *, _QWORD *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "TypeCheckedLoadVCalls",
         0,
         0,
         &v3,
         v4) )
  {
    sub_18839E0(a1, (__int64 *)(a2 + 80));
    (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, v4[0]);
  }
  if ( (!(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 56LL))(a1)
     || *(_QWORD *)(a2 + 112) != *(_QWORD *)(a2 + 104))
    && (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _QWORD, char *, _QWORD *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "TypeTestAssumeConstVCalls",
         0,
         0,
         &v3,
         v4) )
  {
    sub_1883D70(a1, (__int64 *)(a2 + 104));
    (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, v4[0]);
  }
  if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 56LL))(a1)
    || (result = *(_QWORD *)(a2 + 128), *(_QWORD *)(a2 + 136) != result) )
  {
    result = (*(__int64 (__fastcall **)(__int64, const char *, _QWORD, _QWORD, char *, _QWORD *))(*(_QWORD *)a1 + 120LL))(
               a1,
               "TypeCheckedLoadConstVCalls",
               0,
               0,
               &v3,
               v4);
    if ( (_BYTE)result )
    {
      sub_1883D70(a1, (__int64 *)(a2 + 128));
      return (*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, v4[0]);
    }
  }
  return result;
}
