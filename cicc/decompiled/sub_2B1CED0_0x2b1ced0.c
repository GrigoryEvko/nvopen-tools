// Function: sub_2B1CED0
// Address: 0x2b1ced0
//
__int64 __fastcall sub_2B1CED0(__int64 a1, __int64 a2, __int64 a3, int a4)
{
  __int64 v4; // r12
  __int64 v6; // rax
  __int64 v7; // rdi
  __int64 v8; // r14
  __int64 v9; // rbx
  __int64 result; // rax
  unsigned __int8 *v11; // rax
  bool v12; // zf
  __int64 v13; // r14
  __int64 v14; // rax
  int v15; // r15d
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rbx
  __int64 v19; // r13
  __int64 v20; // rdx
  unsigned int v21; // esi
  __int64 v22; // r12
  __int64 v23; // r13
  __int64 v24; // rdx
  unsigned int v25; // esi
  unsigned int v26; // [rsp+0h] [rbp-A0h]
  _BYTE v27[32]; // [rsp+10h] [rbp-90h] BYREF
  __int16 v28; // [rsp+30h] [rbp-70h]
  _BYTE v29[32]; // [rsp+40h] [rbp-60h] BYREF
  __int16 v30; // [rsp+60h] [rbp-40h]

  v4 = a2;
  if ( a4 == 1 )
    return v4;
  switch ( *(_DWORD *)(a1 + 1576) )
  {
    case 0:
    case 2:
    case 0xB:
    case 0x10:
    case 0x11:
    case 0x12:
    case 0x13:
    case 0x14:
      BUG();
    case 1:
      v6 = sub_AD64C0(*(_QWORD *)(a2 + 8), (unsigned int)a4, 0);
      v7 = *(_QWORD *)(a3 + 80);
      v8 = v6;
      v28 = 257;
      v9 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, _QWORD, _QWORD))(*(_QWORD *)v7 + 32LL))(
             v7,
             17,
             a2,
             v6,
             0,
             0);
      if ( !v9 )
      {
        v30 = 257;
        v9 = sub_B504D0(17, a2, v8, (__int64)v29, 0, 0);
        (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(a3 + 88) + 16LL))(
          *(_QWORD *)(a3 + 88),
          v9,
          v27,
          *(_QWORD *)(a3 + 56),
          *(_QWORD *)(a3 + 64));
        v22 = *(_QWORD *)a3;
        v23 = *(_QWORD *)a3 + 16LL * *(unsigned int *)(a3 + 8);
        while ( v23 != v22 )
        {
          v24 = *(_QWORD *)(v22 + 8);
          v25 = *(_DWORD *)v22;
          v22 += 16;
          sub_B99FD0(v9, v25, v24);
        }
      }
      return v9;
    case 3:
    case 4:
    case 6:
    case 7:
    case 8:
    case 9:
    case 0xC:
    case 0xD:
    case 0xE:
    case 0xF:
      return v4;
    case 5:
      if ( (a4 & 1) != 0 )
        return v4;
      result = sub_AD6530(*(_QWORD *)(a2 + 8), a2);
      break;
    case 0xA:
      v11 = sub_AD8DD0(*(_QWORD *)(a2 + 8), (double)a4);
      v12 = *(_BYTE *)(a3 + 108) == 0;
      v28 = 257;
      v13 = (__int64)v11;
      if ( !v12 )
        return sub_B35400(a3, 0x6Cu, a2, (__int64)v11, v26, (__int64)v27, 0, 0, 0);
      v14 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, unsigned __int8 *, _QWORD))(**(_QWORD **)(a3 + 80)
                                                                                           + 40LL))(
              *(_QWORD *)(a3 + 80),
              18,
              a2,
              v11,
              *(unsigned int *)(a3 + 104));
      if ( v14 )
        return v14;
      v30 = 257;
      v15 = *(_DWORD *)(a3 + 104);
      v16 = sub_B504D0(18, a2, v13, (__int64)v29, 0, 0);
      v17 = *(_QWORD *)(a3 + 96);
      v4 = v16;
      if ( v17 )
        sub_B99FD0(v16, 3u, v17);
      sub_B45150(v4, v15);
      (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(a3 + 88) + 16LL))(
        *(_QWORD *)(a3 + 88),
        v4,
        v27,
        *(_QWORD *)(a3 + 56),
        *(_QWORD *)(a3 + 64));
      v18 = *(_QWORD *)a3 + 16LL * *(unsigned int *)(a3 + 8);
      if ( *(_QWORD *)a3 != v18 )
      {
        v19 = *(_QWORD *)a3;
        do
        {
          v20 = *(_QWORD *)(v19 + 8);
          v21 = *(_DWORD *)v19;
          v19 += 16;
          sub_B99FD0(v4, v21, v20);
        }
        while ( v18 != v19 );
      }
      return v4;
    default:
      return 0;
  }
  return result;
}
