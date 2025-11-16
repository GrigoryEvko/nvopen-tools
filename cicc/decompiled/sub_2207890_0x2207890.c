// Function: sub_2207890
// Address: 0x2207890
//
__int64 __fastcall sub_2207890(
        __int64 a1,
        __int64 a2,
        unsigned int a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8)
{
  const char *v12; // r15
  const char *v13; // rsi
  __int64 v14; // rcx
  __int64 v15; // rdx
  char v16; // r8
  int v17; // eax
  const char *v19; // rsi
  int v20; // eax
  char v21; // [rsp+0h] [rbp-58h]
  __int64 v22; // [rsp+0h] [rbp-58h]
  __int64 v23; // [rsp+8h] [rbp-50h]
  __int64 v24; // [rsp+8h] [rbp-50h]
  __int64 v25; // [rsp+10h] [rbp-48h]

  v12 = *(const char **)(a1 + 8);
  v13 = *(const char **)(a4 + 8);
  v14 = a7;
  v15 = a8;
  if ( v12 == v13 )
    goto LABEL_4;
  v16 = *v12;
  if ( *v12 != 42 )
  {
    v23 = a6;
    v21 = *v12;
    v17 = strcmp(*(const char **)(a1 + 8), v13);
    v16 = v21;
    a6 = v23;
    v14 = a7;
    v15 = a8;
    if ( !v17 )
    {
LABEL_4:
      *(_QWORD *)v15 = a5;
      *(_DWORD *)(v15 + 8) = a3;
      if ( a2 < 0 )
      {
        if ( a2 == -2 )
          *(_DWORD *)(v15 + 16) = 1;
      }
      else
      {
        *(_DWORD *)(v15 + 16) = 5 * (v14 == a2 + a5) + 1;
      }
      return 0;
    }
  }
  if ( a5 == v14 )
  {
    v19 = *(const char **)(a6 + 8);
    if ( v12 == v19
      || v16 != 42 && (v25 = v15, v24 = v14, v22 = a6, v20 = strcmp(v12, v19), a6 = v22, v14 = v24, v15 = v25, !v20) )
    {
      *(_DWORD *)(v15 + 12) = a3;
      return 0;
    }
  }
  return (*(__int64 (__fastcall **)(_QWORD, __int64, _QWORD, __int64, __int64, __int64, __int64, __int64))(**(_QWORD **)(a1 + 16) + 56LL))(
           *(_QWORD *)(a1 + 16),
           a2,
           a3,
           a4,
           a5,
           a6,
           v14,
           v15);
}
