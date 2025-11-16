// Function: sub_15DE7A0
// Address: 0x15de7a0
//
__int64 __fastcall sub_15DE7A0(unsigned int **a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rdx
  unsigned int v5; // r15d
  __int64 result; // rax
  __int64 v7; // rdi
  __int64 v8; // rax
  __int64 v9; // rbx
  __int64 v10; // rsi
  __int64 v11; // rax
  int v12; // ebx
  __int64 v13; // r10
  __int64 v14; // rax
  __int64 v15; // rdx
  _BYTE *v16; // rsi
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rsi
  __int64 v20; // [rsp+8h] [rbp-98h]
  __int64 v21; // [rsp+18h] [rbp-88h]
  _BYTE *v22; // [rsp+20h] [rbp-80h] BYREF
  __int64 v23; // [rsp+28h] [rbp-78h]
  _BYTE v24[112]; // [rsp+30h] [rbp-70h] BYREF

  v4 = **a1;
  v5 = (*a1)[1];
  *a1 += 2;
  a1[1] = (unsigned int *)((char *)a1[1] - 1);
  switch ( v4 )
  {
    case 0LL:
    case 1LL:
      return sub_1643270(a3);
    case 2LL:
      return sub_1643310(a3);
    case 3LL:
      return sub_16432D0(a3);
    case 4LL:
      return sub_16432C0(a3);
    case 5LL:
      return sub_1643290(a3);
    case 6LL:
      return sub_16432A0(a3);
    case 7LL:
      return sub_16432B0(a3);
    case 8LL:
      return sub_16432F0(a3);
    case 9LL:
      v19 = v5;
      return sub_1644900(a3, v19);
    case 10LL:
      v18 = sub_15DE7A0(a1, a2, a3);
      return sub_16463B0(v18, v5);
    case 11LL:
      v17 = sub_15DE7A0(a1, a2, a3);
      return sub_1646BA0(v17, v5);
    case 12LL:
      v22 = v24;
      v23 = 0x800000000LL;
      if ( v5 )
      {
        v12 = 0;
        do
        {
          v13 = sub_15DE7A0(a1, a2, a3);
          v14 = (unsigned int)v23;
          if ( (unsigned int)v23 >= HIDWORD(v23) )
          {
            v20 = v13;
            sub_16CD150(&v22, v24, 0, 8);
            v14 = (unsigned int)v23;
            v13 = v20;
          }
          ++v12;
          *(_QWORD *)&v22[8 * v14] = v13;
          v15 = (unsigned int)(v23 + 1);
          LODWORD(v23) = v23 + 1;
        }
        while ( v5 != v12 );
        v16 = v22;
      }
      else
      {
        v16 = v24;
        v15 = 0;
      }
      result = sub_1645600(a3, v16, v15, 0);
      if ( v22 != v24 )
      {
        v21 = result;
        _libc_free((unsigned __int64)v22);
        return v21;
      }
      return result;
    case 13LL:
      return *(_QWORD *)(a2 + 8LL * (v5 >> 3));
    case 14LL:
      v9 = *(_QWORD *)(a2 + 8LL * (v5 >> 3));
      if ( *(_BYTE *)(v9 + 8) == 16 )
      {
        v10 = 2 * (unsigned int)sub_1643030(*(_QWORD *)(v9 + 24));
        goto LABEL_14;
      }
      v19 = (unsigned int)(2 * (*(_DWORD *)(v9 + 8) >> 8));
      return sub_1644900(a3, v19);
    case 15LL:
      v9 = *(_QWORD *)(a2 + 8LL * (v5 >> 3));
      if ( *(_BYTE *)(v9 + 8) == 16 )
      {
        v10 = (unsigned int)sub_1643030(*(_QWORD *)(v9 + 24)) >> 1;
LABEL_14:
        v11 = sub_1644900(*(_QWORD *)v9, v10);
        return sub_16463B0(v11, *(_QWORD *)(v9 + 32));
      }
      else
      {
        v19 = *(_DWORD *)(v9 + 8) >> 9;
        return sub_1644900(a3, v19);
      }
    case 16LL:
      v8 = *(_QWORD *)(a2 + 8LL * (v5 >> 3));
      return sub_16463B0(*(_QWORD *)(v8 + 24), *(_DWORD *)(v8 + 32) >> 1);
    case 17LL:
      v7 = sub_15DE7A0(a1, a2, a3);
      return sub_16463B0(v7, *(_QWORD *)(*(_QWORD *)(a2 + 8LL * (v5 >> 3)) + 32LL));
    case 18LL:
      return sub_1646BA0(*(_QWORD *)(a2 + 8LL * (v5 >> 3)), 0);
    case 19LL:
      return sub_1646BA0(**(_QWORD **)(*(_QWORD *)(a2 + 8LL * (v5 >> 3)) + 16LL), 0);
    case 20LL:
      return *(_QWORD *)(a2 + 8LL * HIWORD(v5));
  }
}
