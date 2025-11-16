// Function: sub_3027FE0
// Address: 0x3027fe0
//
void __fastcall sub_3027FE0(_BYTE **a1, __int64 a2, unsigned int a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rdx
  __int64 v6; // rsi
  __int64 v9; // rax
  int v10; // edi
  const char *v11; // rax
  __int64 v12; // r12
  unsigned int v13; // eax
  unsigned __int8 *v14[2]; // [rsp+0h] [rbp-30h] BYREF
  __int64 v15; // [rsp+10h] [rbp-20h] BYREF

  v5 = 5LL * a3;
  v6 = *(_QWORD *)(a2 + 32) + 8 * v5;
  switch ( *(_BYTE *)v6 )
  {
    case 0:
      v10 = *(_DWORD *)(v6 + 8);
      if ( (unsigned int)(v10 - 1) > 0x3FFFFFFE )
      {
        sub_30232C0((__int64)v14, (__int64)a1, v10);
        sub_CB6200(a4, v14[0], (size_t)v14[1]);
        if ( (__int64 *)v14[0] != &v15 )
          j_j___libc_free_0((unsigned __int64)v14[0]);
      }
      else if ( v10 == 1 )
      {
        v12 = sub_904010(a4, "__local_depot");
        v13 = sub_31DA6A0(a1);
        sub_CB59D0(v12, v13);
      }
      else
      {
        v11 = (const char *)sub_35EE460();
        sub_904010(a4, v11);
      }
      break;
    case 1:
      sub_CB59F0(a4, *(_QWORD *)(v6 + 24));
      break;
    case 3:
      sub_3026430((__int64)a1, *(_QWORD *)(v6 + 24), a4);
      break;
    case 4:
      v9 = sub_2E309C0(*(_QWORD *)(v6 + 24), v6, v5, a4, a5);
      sub_EA12C0(v9, a4, a1[26]);
      break;
    case 0xA:
      (*((void (__fastcall **)(_BYTE **, __int64, __int64))*a1 + 65))(a1, v6, a4);
      break;
    default:
      BUG();
  }
}
