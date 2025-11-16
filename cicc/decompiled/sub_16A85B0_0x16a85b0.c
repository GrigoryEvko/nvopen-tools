// Function: sub_16A85B0
// Address: 0x16a85b0
//
__int64 __fastcall sub_16A85B0(__int64 a1, __int64 a2)
{
  __int64 v4; // rdx
  unsigned __int64 v5; // rax
  unsigned int v6; // eax
  __int64 v7; // rcx
  __int64 v8; // rsi
  unsigned int v9; // esi
  unsigned __int64 v10; // rdi
  unsigned __int64 v12; // rax
  unsigned __int64 v13; // rax
  unsigned __int64 v14; // rax
  unsigned __int64 v15; // rax
  unsigned __int64 v16; // [rsp+0h] [rbp-20h] BYREF
  unsigned int v17; // [rsp+8h] [rbp-18h]

  LODWORD(v4) = *(_DWORD *)(a2 + 8);
  switch ( (_DWORD)v4 )
  {
    case 0x10:
      v12 = *(_QWORD *)a2;
      *(_DWORD *)(a1 + 8) = 16;
      *(_QWORD *)a1 = (unsigned __int16)__ROL2__(v12, 8);
      return a1;
    case 0x20:
      v14 = *(_QWORD *)a2;
      *(_DWORD *)(a1 + 8) = 32;
      *(_QWORD *)a1 = _byteswap_ulong(v14);
      return a1;
    case 0x30:
      v15 = *(_QWORD *)a2;
      *(_DWORD *)(a1 + 8) = 48;
      *(_QWORD *)a1 = _byteswap_ulong(v15 >> 16) | ((unsigned __int64)(unsigned __int16)__ROL2__(v15, 8) << 32);
      return a1;
    case 0x40:
      v13 = *(_QWORD *)a2;
      *(_DWORD *)(a1 + 8) = 64;
      *(_QWORD *)a1 = _byteswap_uint64(v13);
      return a1;
    default:
      LODWORD(v5) = ((unsigned __int64)(unsigned int)v4 + 63) >> 6;
      v17 = (_DWORD)v5 << 6;
      if ( (_DWORD)v5 << 6 <= 0x40u )
      {
        v16 = 0;
      }
      else
      {
        sub_16A4EF0((__int64)&v16, 0, 0);
        v4 = *(unsigned int *)(a2 + 8);
        v5 = (unsigned __int64)(v4 + 63) >> 6;
      }
      if ( (_DWORD)v5 )
      {
        v6 = v5 - 1;
        v7 = 0;
        do
        {
          v8 = v6--;
          *(_QWORD *)(v16 + v7) = _byteswap_uint64(*(_QWORD *)(*(_QWORD *)a2 + 8 * v8));
          v7 += 8;
        }
        while ( v6 != -1 );
        LODWORD(v4) = *(_DWORD *)(a2 + 8);
      }
      if ( v17 != (_DWORD)v4 )
      {
        v9 = v17 - v4;
        if ( v17 <= 0x40 )
        {
          v10 = 0;
          if ( v17 != v9 )
            v10 = v16 >> v9;
          goto LABEL_15;
        }
        sub_16A8110((__int64)&v16, v9);
        LODWORD(v4) = *(_DWORD *)(a2 + 8);
      }
      v10 = v16;
LABEL_15:
      *(_DWORD *)(a1 + 8) = v4;
      *(_QWORD *)a1 = v10;
      return a1;
  }
}
