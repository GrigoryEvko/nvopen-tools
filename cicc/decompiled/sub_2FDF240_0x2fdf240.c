// Function: sub_2FDF240
// Address: 0x2fdf240
//
__int64 __fastcall sub_2FDF240(__int64 a1, _QWORD *a2, __int64 a3, int a4, __int64 a5, int a6)
{
  unsigned __int64 v6; // rax
  __int64 v8; // rcx
  int v11; // edx
  unsigned __int64 v12; // rsi
  __int64 v14; // r10
  int v15; // r8d
  __int64 v16; // rdx
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // [rsp+0h] [rbp-20h]
  __int64 v20; // [rsp+8h] [rbp-18h]

  if ( a2 && (v8 = a2[13]) != 0 )
  {
    v11 = *(_DWORD *)(a3 + 24);
    LOBYTE(v12) = 0;
    if ( v11 < 0 )
    {
      v14 = *(_QWORD *)(a1 + 8);
      v15 = *(_DWORD *)(a5 + 24);
      v16 = *(unsigned __int16 *)(v14 - 40LL * (unsigned int)~v11 + 6);
      if ( v15 < 0 )
      {
        v6 = sub_2FDF1B0(a2, (unsigned __int16)v16, a4, *(unsigned __int16 *)(v14 - 40LL * (unsigned int)~v15 + 6), a6);
        HIDWORD(v20) = HIDWORD(v6);
        v12 = HIDWORD(v6);
      }
      else
      {
        v17 = v8 + 10 * v16;
        v18 = a4 + (unsigned int)*(unsigned __int16 *)(v17 + 6);
        if ( *(unsigned __int16 *)(v17 + 8) > (unsigned int)v18 )
        {
          LOBYTE(v12) = 1;
          LODWORD(v6) = *(_DWORD *)(a2[11] + 4 * v18);
        }
      }
    }
    LODWORD(v20) = v6;
    BYTE4(v20) = v12;
    return v20;
  }
  else
  {
    BYTE4(v19) = 0;
    return v19;
  }
}
