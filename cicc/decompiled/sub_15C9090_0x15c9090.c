// Function: sub_15C9090
// Address: 0x15c9090
//
void __fastcall sub_15C9090(__int64 a1, _QWORD *a2)
{
  bool v2; // zf
  __int64 v4; // rax
  __int64 v5; // rax
  const char *v6; // rdi
  __int64 v7; // rdx

  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  v2 = *a2 == 0;
  *(_QWORD *)(a1 + 16) = 0;
  if ( !v2 )
  {
    v4 = sub_15C70A0((__int64)a2);
    v5 = *(_QWORD *)(v4 - 8LL * *(unsigned int *)(v4 + 8));
    if ( *(_BYTE *)v5 == 15 || (v5 = *(_QWORD *)(v5 - 8LL * *(unsigned int *)(v5 + 8))) != 0 )
    {
      v6 = *(const char **)(v5 - 8LL * *(unsigned int *)(v5 + 8));
      if ( v6 )
        v6 = (const char *)sub_161E970(v6);
      else
        v7 = 0;
    }
    else
    {
      v7 = 0;
      v6 = byte_3F871B3;
    }
    *(_QWORD *)a1 = v6;
    *(_QWORD *)(a1 + 8) = v7;
    *(_DWORD *)(a1 + 16) = *(_DWORD *)(sub_15C70A0((__int64)a2) + 4);
    *(_DWORD *)(a1 + 20) = *(unsigned __int16 *)(sub_15C70A0((__int64)a2) + 2);
  }
}
