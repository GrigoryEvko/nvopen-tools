// Function: sub_1B92390
// Address: 0x1b92390
//
__int64 __fastcall sub_1B92390(_QWORD *a1, __int64 a2, unsigned int a3, __m128i a4, __m128i a5)
{
  __int64 v5; // r13
  __int64 v6; // r12
  __int64 v7; // r15
  int v8; // ebx
  unsigned int v9; // r15d
  __int64 v11; // rbx
  __int64 v12; // rax
  __int64 v13; // r12
  __int64 v14; // r15
  __int64 v15; // r13
  __int64 v16; // r14
  __int64 v17; // rax
  __int64 v18; // [rsp+0h] [rbp-80h]
  __int64 v19; // [rsp+18h] [rbp-68h]
  __int64 v20; // [rsp+38h] [rbp-48h]

  v5 = (__int64)a1;
  v6 = a2;
  sub_1B8DFF0(a2);
  v7 = sub_13A4950(a2);
  sub_1B8E090(*(__int64 **)v7, a3);
  if ( *(_BYTE *)(v7 + 16) == 56 )
  {
    v18 = a1[38];
    v11 = *(_QWORD *)(v18 + 112);
    v12 = *(_DWORD *)(v7 + 20) & 0xFFFFFFF;
    if ( (unsigned int)v12 > 1 )
    {
      v13 = v7;
      v14 = 1;
      v19 = a1[40];
      v15 = a1[37];
      v20 = (unsigned int)(v12 - 1);
      while ( 1 )
      {
        v16 = *(_QWORD *)(v13 + 24 * (v14 - v12));
        v17 = sub_146F1B0(v11, v16);
        if ( !sub_146CEE0(v11, v17, v15) && !(unsigned __int8)sub_1BF28C0(v19, v16) )
        {
          v5 = (__int64)a1;
          v6 = a2;
          goto LABEL_2;
        }
        if ( v14 == v20 )
          break;
        ++v14;
        v12 = *(_DWORD *)(v13 + 20) & 0xFFFFFFF;
      }
      v7 = v13;
      v5 = (__int64)a1;
      v6 = a2;
    }
    sub_1494E70(v18, v7, a4, a5);
  }
LABEL_2:
  v8 = sub_14A3620(*(_QWORD *)(v5 + 328));
  v9 = a3 * (sub_14A34A0(*(_QWORD *)(v5 + 328)) + v8);
  if ( a3 != 1 )
    v9 += sub_1B8FA60(v6, a3, *(__int64 **)(v5 + 328));
  if ( (unsigned __int8)sub_1B91FD0(v5, v6) )
  {
    v9 >>= 1;
    if ( sub_1B92360((_DWORD *)v5, v6) )
      return 3000000;
  }
  return v9;
}
