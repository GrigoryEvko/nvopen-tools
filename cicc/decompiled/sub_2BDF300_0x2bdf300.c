// Function: sub_2BDF300
// Address: 0x2bdf300
//
unsigned __int64 *__fastcall sub_2BDF300(__int64 a1)
{
  unsigned __int8 *v1; // rax
  unsigned int v2; // r14d
  __int64 v3; // r13
  __int64 v4; // r15
  unsigned __int8 v5; // bl
  int v6; // esi
  __int64 (__fastcall *v7)(__int64, unsigned int); // rcx
  signed __int8 v8; // al
  char *v9; // rax
  int v10; // eax
  unsigned __int64 *result; // rax

  v1 = *(unsigned __int8 **)(a1 + 176);
  if ( v1 == *(unsigned __int8 **)(a1 + 184) )
    goto LABEL_17;
  v2 = (char)*v1;
  v3 = *(_QWORD *)(a1 + 192);
  v4 = *v1;
  v5 = *v1;
  v6 = *(char *)(v3 + v4 + 313);
  if ( !*(_BYTE *)(v3 + v4 + 313) )
  {
    v6 = (char)*v1;
    v7 = *(__int64 (__fastcall **)(__int64, unsigned int))(*(_QWORD *)v3 + 64LL);
    v8 = *v1;
    if ( v7 != sub_2216C50 )
    {
      v8 = ((__int64 (__fastcall *)(_QWORD, _QWORD, _QWORD))v7)(*(_QWORD *)(a1 + 192), v2, 0);
      v6 = v8;
    }
    if ( v8 )
      *(_BYTE *)(v3 + v4 + 313) = v8;
  }
  v9 = strchr(*(const char **)(a1 + 160), v6);
  if ( v9 && *v9 )
  {
    *(_DWORD *)(a1 + 144) = 1;
    goto LABEL_14;
  }
  v10 = *(_DWORD *)(a1 + 140);
  if ( (v10 & 0x80u) == 0 )
  {
    if ( (v10 & 0x120) != 0
      && (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)(a1 + 192) + 48LL) + 2LL * v5 + 1) & 8) != 0
      && v5 != 48 )
    {
      *(_DWORD *)(a1 + 144) = 4;
LABEL_14:
      result = sub_2240FD0((unsigned __int64 *)(a1 + 200), 0, *(_QWORD *)(a1 + 208), 1u, v2);
      ++*(_QWORD *)(a1 + 176);
      return result;
    }
LABEL_17:
    abort();
  }
  return sub_2BDF0D0(a1);
}
