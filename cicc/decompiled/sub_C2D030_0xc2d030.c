// Function: sub_C2D030
// Address: 0xc2d030
//
__int64 __fastcall sub_C2D030(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rdx
  __int64 (__fastcall *v6)(__int64); // rax
  __int64 result; // rax
  __int64 v8; // rax
  unsigned __int64 v9; // rdx
  unsigned __int8 v10; // si
  __int64 v11; // rdx
  int v12; // eax
  unsigned __int8 v13; // si

  v4 = a2 + a3;
  *(_QWORD *)(a1 + 208) = a2;
  *(_QWORD *)(a1 + 216) = v4;
  switch ( *(_DWORD *)a4 )
  {
    case 1:
      result = sub_C24440((const __m128i **)a1);
      if ( (_DWORD)result )
        return result;
      if ( *(_DWORD *)a4 != 1 )
        goto LABEL_37;
      v8 = *(_QWORD *)(a4 + 8);
      if ( (v8 & 0x100000000LL) == 0 )
        goto LABEL_11;
      *(_BYTE *)(*(_QWORD *)(a1 + 80) + 72LL) = 1;
      if ( *(_DWORD *)a4 != 1 )
        goto LABEL_37;
      v8 = *(_QWORD *)(a4 + 8);
LABEL_11:
      if ( (v8 & 0x200000000LL) == 0 )
        goto LABEL_14;
      *(_BYTE *)(a1 + 178) = 1;
      unk_4F838D3 = 1;
      if ( *(_DWORD *)a4 != 1 )
        goto LABEL_37;
      v8 = *(_QWORD *)(a4 + 8);
LABEL_14:
      if ( (v8 & 0x1000000000LL) != 0 )
      {
        *(_BYTE *)(a1 + 179) = 1;
        unk_4F838D2 = 1;
        if ( *(_DWORD *)a4 != 1 )
LABEL_37:
          BUG();
        v8 = *(_QWORD *)(a4 + 8);
      }
      if ( (v8 & 0x400000000LL) != 0 )
      {
        *(_BYTE *)(a1 + 184) = 1;
        unk_4F838D0 = 1;
      }
      goto LABEL_4;
    case 2:
      v9 = *(_QWORD *)(a4 + 8);
      v10 = BYTE4(v9) & 1;
      *(_BYTE *)(a1 + 204) |= BYTE4(v9) & 1;
      v11 = (v9 >> 33) & 1;
      if ( *(_DWORD *)a4 != 2 )
        goto LABEL_37;
      unk_4C5C708 = (*(_QWORD *)(a4 + 8) & 0x400000000LL) != 0;
      result = sub_C23390(a1, v10, v11);
      if ( (_DWORD)result )
        return result;
      goto LABEL_4;
    case 3:
      result = sub_C20C80((__int64 *)a1, a2);
      if ( (_DWORD)result )
        return result;
      goto LABEL_4;
    case 4:
      if ( !*(_QWORD *)(a1 + 192) )
      {
        *(_QWORD *)(a1 + 208) = v4;
        goto LABEL_4;
      }
      result = sub_C26350(a1);
      if ( !(_DWORD)result )
        goto LABEL_4;
      return result;
    case 5:
      v12 = *(_DWORD *)(a4 + 12) & 1;
      *(_BYTE *)(a1 + 177) = *(_BYTE *)(a4 + 12) & 1;
      unk_4F838D4 = v12;
      if ( *(_DWORD *)a4 != 5 )
        goto LABEL_37;
      v13 = (*(_QWORD *)(a4 + 8) & 0x200000000LL) != 0;
      *(_BYTE *)(a1 + 176) = v13;
      result = sub_C28B20((_QWORD *)a1, v13);
      if ( (_DWORD)result )
        return result;
      goto LABEL_4;
    case 6:
      result = sub_C23AD0((_QWORD *)a1);
      if ( (_DWORD)result )
        return result;
      goto LABEL_4;
    case 0x20:
      *(_QWORD *)(a1 + 160) = a2;
      *(_QWORD *)(a1 + 168) = v4;
      result = sub_C2CF70((_QWORD *)a1);
      if ( (_DWORD)result )
        return result;
      goto LABEL_4;
    default:
      v6 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 104LL);
      if ( v6 == sub_C1E960 )
      {
        *(_QWORD *)(a1 + 208) = v4;
        sub_C1AFD0();
LABEL_4:
        sub_C1AFD0();
        return 0;
      }
      result = ((__int64 (__fastcall *)(__int64, __int64))v6)(a1, a4);
      if ( !(_DWORD)result )
        goto LABEL_4;
      return result;
  }
}
