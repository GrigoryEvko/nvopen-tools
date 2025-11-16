// Function: sub_2E8E4C0
// Address: 0x2e8e4c0
//
unsigned __int64 __fastcall sub_2E8E4C0(__int64 a1, __int64 a2)
{
  __int64 (*v2)(); // rax
  unsigned __int64 result; // rax
  __int64 v4; // rax
  __int64 v5; // rax
  unsigned __int64 v6; // rdx
  int v7; // eax
  unsigned __int64 v8; // rcx
  char v9; // al
  unsigned __int64 v10; // rdx
  char v11; // di
  int v12; // esi
  unsigned __int64 v13; // rax
  unsigned __int64 v14; // rcx
  unsigned __int64 v15; // rax
  unsigned __int64 v16; // rax
  int v17; // [rsp+1Ch] [rbp-24h] BYREF
  unsigned __int64 v18; // [rsp+20h] [rbp-20h]
  __int64 v19; // [rsp+28h] [rbp-18h]

  v2 = *(__int64 (**)())(*(_QWORD *)a2 + 104LL);
  if ( v2 == sub_2E85440
    || !((unsigned int (__fastcall *)(__int64, __int64, int *))v2)(a2, a1, &v17)
    || (v4 = sub_2E88D60(a1),
        !*(_BYTE *)(*(_QWORD *)(*(_QWORD *)(v4 + 48) + 8LL)
                  + 40LL * (unsigned int)(v17 + *(_DWORD *)(*(_QWORD *)(v4 + 48) + 32LL))
                  + 18)) )
  {
    LOBYTE(v19) = 0;
    return v18;
  }
  v5 = *(_QWORD *)(a1 + 48);
  v6 = v5 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (v5 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
    goto LABEL_21;
  v7 = v5 & 7;
  if ( !v7 )
  {
    *(_QWORD *)(a1 + 48) = v6;
    goto LABEL_9;
  }
  if ( v7 != 3 )
LABEL_21:
    BUG();
  v6 = *(_QWORD *)(v6 + 16);
LABEL_9:
  v8 = *(_QWORD *)(v6 + 24);
  result = -1;
  if ( (v8 & 0xFFFFFFFFFFFFFFF9LL) != 0 )
  {
    v9 = *(_BYTE *)(v6 + 24);
    v10 = v8 >> 3;
    v11 = v9 & 2;
    if ( (v9 & 6) == 2 || (v9 & 1) != 0 )
    {
      v16 = HIWORD(v8);
      if ( !v11 )
        v16 = HIDWORD(v8);
      result = (v16 + 7) >> 3;
    }
    else
    {
      v12 = (unsigned __int16)((unsigned int)v8 >> 8);
      v13 = v8;
      v14 = HIDWORD(v8);
      v15 = HIWORD(v13);
      if ( !v11 )
        LODWORD(v15) = v14;
      result = ((unsigned __int64)(unsigned int)(v12 * v15) + 7) >> 3;
      if ( (v10 & 1) != 0 )
        result |= 0x4000000000000000uLL;
    }
  }
  v18 = result;
  LOBYTE(v19) = 1;
  return result;
}
