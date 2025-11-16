// Function: sub_893600
// Address: 0x893600
//
__int64 **__fastcall sub_893600(__int64 a1, __int64 **a2, __int64 a3, int a4)
{
  __int64 **v6; // r12
  __int64 **result; // rax
  __int64 *v8; // rbx
  __int64 v9; // r13
  __int64 v10; // rdi
  __int64 v11; // rsi
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 *v15; // r9
  __int64 v16; // rdx
  __int64 v17; // r8
  __int64 *v18; // r9
  __int64 v19; // rax
  __int64 v20; // rsi
  __int64 v21; // [rsp+0h] [rbp-40h]
  _BOOL4 v23; // [rsp+Ch] [rbp-34h]

  v6 = a2;
  switch ( *(_BYTE *)(a1 + 80) )
  {
    case 4:
    case 5:
      v21 = *(_QWORD *)(*(_QWORD *)(a1 + 96) + 80LL);
      break;
    case 6:
      v21 = *(_QWORD *)(*(_QWORD *)(a1 + 96) + 32LL);
      break;
    case 9:
    case 0xA:
      v21 = *(_QWORD *)(*(_QWORD *)(a1 + 96) + 56LL);
      break;
    case 0x13:
    case 0x14:
    case 0x15:
    case 0x16:
      v21 = *(_QWORD *)(a1 + 88);
      break;
    default:
      BUG();
  }
  result = (__int64 **)v21;
  v8 = *(__int64 **)(v21 + 176);
  v9 = *v8;
  if ( a2 )
  {
    do
    {
      while ( 1 )
      {
        v10 = (__int64)v6[5];
        if ( v10 )
        {
          v23 = sub_864700(v10, 0, (__int64)v8, v9, a1, v8[30], 1, 2u);
          sub_8600D0(1u, -1, v8[19], 0);
          if ( a3 )
            sub_886000(a3);
          if ( !unk_4D04828 )
            *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 12) &= ~2u;
          sub_7BC160((__int64)(v6 + 1));
          *((_BYTE *)v6[6] + 32) |= 0x20u;
          sub_6794F0((__int64 **)v6[6], v9, 0);
          *((_BYTE *)v6[6] + 32) &= ~0x20u;
          v11 = (__int64)v6[6];
          sub_8CA950(v8, v11);
          result = (__int64 **)sub_863FC0((__int64)v8, v11, v12, v13, v14, v15);
          if ( v23 )
            break;
        }
        v6 = (__int64 **)*v6;
        if ( !v6 )
          goto LABEL_13;
      }
      result = (__int64 **)sub_863FE0((__int64)v8, v11, v16, v23, v17, v18);
      v6 = (__int64 **)*v6;
    }
    while ( v6 );
LABEL_13:
    if ( a4 )
    {
      result = (__int64 **)dword_4F07590;
      if ( dword_4F07590 )
      {
        v19 = sub_86A2A0(*(_QWORD *)(v21 + 176));
        if ( v19 && *(_BYTE *)(v19 + 16) == 53 )
        {
          result = *(__int64 ***)(v19 + 24);
          v20 = (__int64)result[4];
        }
        else
        {
          result = (__int64 **)v21;
          v20 = *(_QWORD *)(v21 + 264);
        }
        if ( v20 )
          return sub_73BC00(*(_QWORD *)(*(_QWORD *)(v21 + 176) + 152LL), v20);
      }
    }
  }
  return result;
}
