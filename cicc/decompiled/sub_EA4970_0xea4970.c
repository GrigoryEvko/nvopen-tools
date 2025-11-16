// Function: sub_EA4970
// Address: 0xea4970
//
__int64 __fastcall sub_EA4970(__int64 a1, __int64 a2, unsigned __int16 a3)
{
  unsigned int v3; // r13d
  __int64 v6; // rdi
  __int64 (*v7)(); // rax
  __int64 result; // rax
  __int64 v9; // r14
  __int64 v10; // rdx
  __int64 v11; // rsi
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // rdx
  __int128 v16; // [rsp+0h] [rbp-C0h] BYREF
  __int64 v17; // [rsp+10h] [rbp-B0h]
  __int64 v18; // [rsp+18h] [rbp-A8h]
  __int64 v19; // [rsp+20h] [rbp-A0h]
  __int128 v20; // [rsp+30h] [rbp-90h]
  char v21; // [rsp+50h] [rbp-70h]
  char v22; // [rsp+51h] [rbp-6Fh]
  _OWORD v23[2]; // [rsp+60h] [rbp-60h] BYREF
  char v24; // [rsp+80h] [rbp-40h]
  char v25; // [rsp+81h] [rbp-3Fh]

  v3 = a3;
  v6 = *(_QWORD *)(a1 + 8);
  v7 = *(__int64 (**)())(*(_QWORD *)v6 + 168LL);
  if ( v7 == sub_EA21F0
    || (result = ((__int64 (__fastcall *)(__int64, __int64, _QWORD, _QWORD))v7)(v6, a2, a3, *(_QWORD *)(a1 + 224))) == 0 )
  {
    switch ( *(_BYTE *)a2 )
    {
      case 0:
        v9 = sub_EA4970(a1, *(_QWORD *)(a2 + 16), v3);
        v10 = sub_EA4970(a1, *(_QWORD *)(a2 + 24), v3);
        if ( !(v10 | v9) )
          goto LABEL_3;
        if ( v9 )
        {
          if ( !v10 )
            v10 = *(_QWORD *)(a2 + 24);
        }
        else
        {
          v9 = *(_QWORD *)(a2 + 16);
        }
        result = sub_E81A00(*(_DWORD *)a2 >> 8, v9, v10, *(_QWORD **)(a1 + 224), 0);
        break;
      case 1:
      case 4:
        goto LABEL_3;
      case 2:
        if ( (*(_DWORD *)a2 & 0xFFFF00) == 0 )
          return sub_E808D0(*(_QWORD *)(a2 + 16), v3, *(_QWORD **)(a1 + 224), 0);
        v22 = 1;
        *(_QWORD *)&v20 = "' (already modified)";
        v21 = 3;
        v12 = sub_ECD7B0(a1);
        if ( *(_DWORD *)v12 == 2 )
        {
          v14 = *(_QWORD *)(v12 + 8);
          v13 = *(_QWORD *)(v12 + 16);
        }
        else
        {
          v13 = *(_QWORD *)(v12 + 16);
          v14 = *(_QWORD *)(v12 + 8);
          if ( v13 )
          {
            v15 = v13 - 1;
            if ( !v15 )
              v15 = 1;
            ++v14;
            v13 = v15 - 1;
          }
        }
        v17 = v14;
        *(_QWORD *)&v16 = "invalid variant on expression '";
        LOWORD(v19) = 1283;
        v18 = v13;
        v23[1] = v20;
        *(_QWORD *)&v23[0] = &v16;
        v24 = 2;
        v25 = v21;
        sub_ECE0E0(a1, v23, 0, 0);
        return a2;
      case 3:
        v11 = sub_EA4970(a1, *(_QWORD *)(a2 + 16), v3);
        if ( v11 )
          result = sub_E81970(*(_DWORD *)a2 >> 8, v11, *(_QWORD **)(a1 + 224), 0);
        else
LABEL_3:
          result = 0;
        break;
      default:
        BUG();
    }
  }
  return result;
}
