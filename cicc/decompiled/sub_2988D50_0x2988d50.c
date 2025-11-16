// Function: sub_2988D50
// Address: 0x2988d50
//
__int64 __fastcall sub_2988D50(__int64 a1)
{
  signed __int64 v1; // rbx
  __int64 result; // rax
  unsigned __int64 v3; // rbx
  __int64 (__fastcall *v4)(_QWORD *); // r13
  unsigned __int64 v5; // rsi
  _QWORD *v6; // rax
  __int64 v7; // rdx
  unsigned int v8; // esi
  unsigned int v9; // r13d
  _QWORD *v10; // r15
  int v11; // eax
  unsigned __int64 v12; // rax
  unsigned __int64 v13; // rdx
  _QWORD v14[8]; // [rsp+0h] [rbp-40h] BYREF

  v1 = *(_QWORD *)a1;
LABEL_2:
  result = (v1 >> 1) & 3;
  if ( ((v1 >> 1) & 3) != 0 )
  {
LABEL_3:
    if ( (_DWORD)result != ((*(__int64 *)(a1 + 32) >> 1) & 3) )
    {
      v3 = v1 & 0xFFFFFFFFFFFFFFF8LL;
      v4 = *(__int64 (__fastcall **)(_QWORD *))(a1 + 64);
      v5 = *(_QWORD *)(v3 + 32);
      goto LABEL_5;
    }
  }
  else
  {
    while ( 1 )
    {
      v8 = *(_DWORD *)(a1 + 16);
      if ( v8 == *(_DWORD *)(a1 + 48) )
        break;
      v4 = *(__int64 (__fastcall **)(_QWORD *))(a1 + 64);
      v3 = v1 & 0xFFFFFFFFFFFFFFF8LL;
      v5 = sub_B46EC0(*(_QWORD *)(a1 + 8), v8);
LABEL_5:
      v6 = sub_22DE030(*(_QWORD **)(v3 + 8), v5);
      v7 = *(_QWORD *)(a1 + 24);
      v14[0] = v6;
      v14[1] = v7;
      result = v4(v14);
      if ( (_BYTE)result )
        break;
      v1 = *(_QWORD *)a1;
      if ( (*(_QWORD *)a1 & 6) == 0 )
      {
        v9 = *(_DWORD *)(a1 + 16);
        v10 = (_QWORD *)(v1 & 0xFFFFFFFFFFFFFFF8LL);
        do
        {
          *(_DWORD *)(a1 + 16) = ++v9;
          v12 = *v10 & 0xFFFFFFFFFFFFFFF8LL;
          v13 = *(_QWORD *)(v12 + 48) & 0xFFFFFFFFFFFFFFF8LL;
          if ( v13 == v12 + 48 )
            goto LABEL_18;
          if ( !v13 )
            BUG();
          if ( (unsigned int)*(unsigned __int8 *)(v13 - 24) - 30 > 0xA )
LABEL_18:
            v11 = 0;
          else
            v11 = sub_B46E30(v13 - 24);
        }
        while ( v9 != v11 && *(_QWORD *)(v10[1] + 32LL) == sub_B46EC0(*(_QWORD *)(a1 + 8), v9) );
        goto LABEL_2;
      }
      v1 = v1 & 0xFFFFFFFFFFFFFFF9LL | 4;
      *(_QWORD *)a1 = v1;
      result = (v1 >> 1) & 3;
      if ( ((v1 >> 1) & 3) != 0 )
        goto LABEL_3;
    }
  }
  return result;
}
