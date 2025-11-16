// Function: sub_94A4F0
// Address: 0x94a4f0
//
__int64 __fastcall sub_94A4F0(__int64 *a1, __int64 a2, char a3, __int64 a4)
{
  __int64 v6; // rdi
  __int64 v7; // r15
  __int64 v8; // rdi
  __int64 (__fastcall *v9)(__int64, __int64, __int64); // rax
  __int64 v10; // r12
  __int64 v11; // rax
  unsigned int *v12; // rbx
  unsigned int *v13; // r13
  __int64 v14; // rdx
  __int64 v15; // rsi
  __int64 v16; // rdi
  unsigned int v18[8]; // [rsp+0h] [rbp-90h] BYREF
  __int16 v19; // [rsp+20h] [rbp-70h]
  _BYTE v20[32]; // [rsp+30h] [rbp-60h] BYREF
  __int16 v21; // [rsp+50h] [rbp-40h]

  v6 = *(_QWORD *)(a2 + 8);
  if ( a3 )
  {
    v7 = *a1;
    v19 = 257;
    if ( a4 == v6 )
      return a2;
    v8 = *(_QWORD *)(v7 + 128);
    v9 = *(__int64 (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v8 + 120LL);
    if ( (char *)v9 == (char *)sub_920130 )
    {
      if ( *(_BYTE *)a2 > 0x15u )
      {
LABEL_8:
        v21 = 257;
        v11 = sub_BD2C40(72, unk_3F10A14);
        v10 = v11;
        if ( v11 )
          sub_B515B0(v11, a2, a4, v20, 0, 0);
        (*(void (__fastcall **)(_QWORD, __int64, unsigned int *, _QWORD, _QWORD))(**(_QWORD **)(v7 + 136) + 16LL))(
          *(_QWORD *)(v7 + 136),
          v10,
          v18,
          *(_QWORD *)(v7 + 104),
          *(_QWORD *)(v7 + 112));
        v12 = *(unsigned int **)(v7 + 48);
        v13 = &v12[4 * *(unsigned int *)(v7 + 56)];
        while ( v13 != v12 )
        {
          v14 = *((_QWORD *)v12 + 1);
          v15 = *v12;
          v12 += 4;
          sub_B99FD0(v10, v15, v14);
        }
        return v10;
      }
      if ( (unsigned __int8)sub_AC4810(39) )
        v10 = sub_ADAB70(39, a2, a4, 0);
      else
        v10 = sub_AA93C0(39, a2, a4);
    }
    else
    {
      v10 = v9(v8, 39, a2);
    }
    if ( v10 )
      return v10;
    goto LABEL_8;
  }
  if ( !(unsigned __int8)sub_BCAF30(v6, a4) )
    sub_91B8A0("unexpected: cannot convert return value to return type!", (_DWORD *)(*(_QWORD *)a1[1] + 36LL), 1);
  v16 = *a1;
  v21 = 257;
  return sub_949E90((unsigned int **)(v16 + 48), 0x31u, a2, a4, (__int64)v20, 0, v18[0], 0);
}
