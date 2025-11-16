// Function: sub_7E1520
// Address: 0x7e1520
//
__int64 __fastcall sub_7E1520(__int64 a1, const char *a2, __int64 a3, unsigned __int64 a4, __int64 a5)
{
  size_t v6; // rbx
  const char *v7; // r14
  size_t v8; // rax
  char *v9; // r12
  _QWORD *v10; // rcx
  int v11; // eax
  __int64 result; // rax

  v6 = strlen(a2);
  v7 = (const char *)sub_810B00(a1);
  v8 = strlen(v7);
  v9 = (char *)sub_7E1510(v6 + v8 + 1);
  strcpy(v9, a2);
  strcpy(&v9[v6], v7);
  v10 = sub_7DF750((__int64)v9, a3, a4, a5, 1);
  if ( unk_4F068A8
    && (unsigned __int8)(*(_BYTE *)(a3 + 140) - 9) <= 2u
    && (*(_BYTE *)(*(_QWORD *)(a3 + 168) + 109LL) & 8) != 0 )
  {
    v11 = *((unsigned __int8 *)v10 + 145);
    if ( *(_QWORD *)(a3 + 128) % (unsigned __int64)*(unsigned int *)(a3 + 136) )
    {
      v11 |= 0x10u;
      *((_BYTE *)v10 + 145) = v11;
    }
  }
  else
  {
    v11 = *((unsigned __int8 *)v10 + 145);
  }
  result = v11 | 8u;
  *((_BYTE *)v10 + 145) = result;
  return result;
}
