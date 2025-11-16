// Function: sub_C496B0
// Address: 0xc496b0
//
__int64 __fastcall sub_C496B0(__int64 a1, __int64 a2)
{
  __int64 v4; // rax
  int v5; // edi
  unsigned __int64 v6; // rsi
  unsigned int v7; // eax
  unsigned __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // rax
  unsigned __int64 v11; // rdi
  unsigned __int64 v13; // rdx
  char v14; // cl
  unsigned __int64 v15; // rax
  unsigned int v16; // esi
  unsigned __int64 v17; // rax
  unsigned __int64 v18; // [rsp+0h] [rbp-20h] BYREF
  unsigned int v19; // [rsp+8h] [rbp-18h]

  v4 = *(unsigned int *)(a2 + 8);
  if ( (_DWORD)v4 == 16 )
  {
    v15 = *(_QWORD *)a2;
    *(_DWORD *)(a1 + 8) = 16;
    *(_QWORD *)a1 = (unsigned __int16)__ROL2__(v15, 8);
    return a1;
  }
  else if ( (_DWORD)v4 == 32 )
  {
    v17 = *(_QWORD *)a2;
    *(_DWORD *)(a1 + 8) = 32;
    *(_QWORD *)a1 = _byteswap_ulong(v17);
    return a1;
  }
  else
  {
    if ( (unsigned int)v4 > 0x40 )
    {
      v5 = (unsigned __int64)(v4 + 63) >> 6;
      v19 = v5 << 6;
      if ( (unsigned int)(v5 << 6) > 0x40 )
      {
        sub_C43690((__int64)&v18, 0, 0);
        v10 = *(unsigned int *)(a2 + 8);
        v5 = (unsigned __int64)(v10 + 63) >> 6;
        if ( !((unsigned __int64)(v10 + 63) >> 6) )
          goto LABEL_10;
        v6 = v18;
      }
      else
      {
        v18 = 0;
        v6 = 0;
      }
      v7 = 0;
      while ( 1 )
      {
        v8 = *(_QWORD *)(*(_QWORD *)a2 + 8LL * (v5 - 1 - v7));
        v9 = v7++;
        *(_QWORD *)(v6 + 8 * v9) = _byteswap_uint64(v8);
        if ( v7 == v5 )
          break;
        v6 = v18;
      }
      LODWORD(v10) = *(_DWORD *)(a2 + 8);
LABEL_10:
      if ( v19 != (_DWORD)v10 )
      {
        v16 = v19 - v10;
        if ( v19 <= 0x40 )
        {
          v11 = 0;
          if ( v19 != v16 )
            v11 = v18 >> v16;
          goto LABEL_12;
        }
        sub_C482E0((__int64)&v18, v16);
        LODWORD(v10) = *(_DWORD *)(a2 + 8);
      }
      v11 = v18;
LABEL_12:
      *(_DWORD *)(a1 + 8) = v10;
      *(_QWORD *)a1 = v11;
      return a1;
    }
    v13 = *(_QWORD *)a2;
    *(_DWORD *)(a1 + 8) = v4;
    v14 = 64 - v4;
    *(_QWORD *)a1 = _byteswap_uint64(v13) >> v14;
    return a1;
  }
}
