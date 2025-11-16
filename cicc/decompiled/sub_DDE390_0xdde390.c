// Function: sub_DDE390
// Address: 0xdde390
//
__int64 __fastcall sub_DDE390(
        __int64 a1,
        __int64 *a2,
        unsigned __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7)
{
  _BYTE *v8; // r13
  bool v10; // al
  __int64 v11; // r9
  unsigned int v13; // eax
  _BYTE *v14; // rax
  __int64 v15; // rax
  __int64 v16; // r9
  unsigned __int64 v17; // rdx
  __int64 v18; // rdx
  __int64 v19; // rcx
  unsigned int v20; // ebx
  __int64 v21; // r8
  __int64 v22; // rax
  __int64 v23; // rax
  unsigned int v24; // eax
  unsigned __int64 v25; // [rsp+8h] [rbp-58h]
  unsigned int v26; // [rsp+14h] [rbp-4Ch]
  __int64 v28; // [rsp+18h] [rbp-48h]

  v8 = (_BYTE *)a5;
  v26 = a3;
  v25 = HIDWORD(a3);
  v10 = sub_DADE90((__int64)a2, a5, a6);
  v11 = a6;
  if ( !v10 )
  {
    if ( !sub_DADE90((__int64)a2, a4, a6) )
      goto LABEL_4;
    v13 = sub_B52F50(a3);
    v11 = a6;
    v26 = v13;
    v14 = (_BYTE *)a4;
    a4 = (__int64)v8;
    v8 = v14;
  }
  if ( *(_WORD *)(a4 + 24) == 8 && v11 == *(_QWORD *)(a4 + 48) )
  {
    v28 = v11;
    v15 = sub_DC1950((__int64)a2, a4, v26);
    v16 = v28;
    if ( BYTE4(v15) )
    {
      if ( (_DWORD)v15 )
      {
        v24 = sub_B52870(v26);
        v16 = v28;
        v17 = ((unsigned __int64)(unsigned __int8)v25 << 32) | v24;
      }
      else
      {
        v17 = a3 & 0xFFFFFFFF00000000LL | v26;
      }
      if ( (unsigned __int8)sub_DDDA00((__int64)a2, v16, v17, a4, v8)
        || a7
        && v26 - 36 <= 1
        && (v20 = sub_B53550(v26), (*(_BYTE *)(a4 + 28) & 4) != 0)
        && *(_QWORD *)(a4 + 40) == 2
        && (v22 = sub_D33D80((_QWORD *)a4, (__int64)a2, v18, v19, v21), (unsigned __int8)sub_DBEDC0((__int64)a2, v22))
        && (unsigned __int8)sub_DBED40((__int64)a2, (__int64)v8)
        && (unsigned __int8)sub_DDCB50(a2, v20, (_BYTE *)a4, v8, a7) )
      {
        v23 = **(_QWORD **)(a4 + 32);
        *(_DWORD *)a1 = v26;
        *(_QWORD *)(a1 + 16) = v8;
        *(_QWORD *)(a1 + 8) = v23;
        *(_BYTE *)(a1 + 4) = v25;
        *(_BYTE *)(a1 + 24) = 1;
        return a1;
      }
    }
  }
LABEL_4:
  *(_BYTE *)(a1 + 24) = 0;
  return a1;
}
