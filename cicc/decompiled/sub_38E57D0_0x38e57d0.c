// Function: sub_38E57D0
// Address: 0x38e57d0
//
__int64 __fastcall sub_38E57D0(__int64 a1, __int64 a2, unsigned __int16 a3)
{
  unsigned int v3; // r13d
  __int64 v6; // rdi
  __int64 (*v7)(); // rax
  __int64 result; // rax
  __int64 v9; // r14
  __int64 v10; // rdx
  __int64 v11; // rsi
  __int64 v12; // rax
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // rdx
  unsigned __int64 v16; // rax
  unsigned __int64 v17; // rcx
  unsigned __int64 v18; // rcx
  unsigned __int64 v19; // rax
  __int64 v20; // [rsp+0h] [rbp-90h] BYREF
  __int128 v21; // [rsp+10h] [rbp-80h] BYREF
  __int64 v22; // [rsp+20h] [rbp-70h]
  const char *v23; // [rsp+30h] [rbp-60h]
  char v24; // [rsp+40h] [rbp-50h]
  char v25; // [rsp+41h] [rbp-4Fh]
  _QWORD v26[2]; // [rsp+50h] [rbp-40h] BYREF
  char v27; // [rsp+60h] [rbp-30h]
  char v28; // [rsp+61h] [rbp-2Fh]

  v3 = a3;
  v6 = *(_QWORD *)(a1 + 8);
  v7 = *(__int64 (**)())(*(_QWORD *)v6 + 152LL);
  if ( v7 == sub_38E2A10
    || (result = ((__int64 (__fastcall *)(__int64, __int64, _QWORD, _QWORD))v7)(v6, a2, a3, *(_QWORD *)(a1 + 320))) == 0 )
  {
    switch ( *(_DWORD *)a2 )
    {
      case 0:
        v9 = sub_38E57D0(a1, *(_QWORD *)(a2 + 24), v3);
        v10 = sub_38E57D0(a1, *(_QWORD *)(a2 + 32), v3);
        if ( !(v10 | v9) )
          goto LABEL_3;
        if ( v9 )
        {
          if ( !v10 )
            v10 = *(_QWORD *)(a2 + 32);
        }
        else
        {
          v9 = *(_QWORD *)(a2 + 24);
        }
        result = sub_38CB1F0(*(_DWORD *)(a2 + 16), v9, v10, *(_QWORD *)(a1 + 320), 0);
        break;
      case 1:
      case 4:
        goto LABEL_3;
      case 2:
        if ( *(_WORD *)(a2 + 16) )
        {
          v25 = 1;
          v23 = "' (already modified)";
          v24 = 3;
          v12 = sub_3909460(a1);
          v15 = v12;
          if ( *(_DWORD *)v12 == 2 )
          {
            v19 = *(_QWORD *)(v12 + 8);
            v17 = *(_QWORD *)(v15 + 16);
          }
          else
          {
            v16 = *(_QWORD *)(v12 + 16);
            v17 = 0;
            if ( v16 )
            {
              v18 = v16 - 1;
              if ( v16 == 1 )
                v18 = 1;
              if ( v18 > v16 )
                v18 = v16;
              v16 = 1;
              v17 = v18 - 1;
            }
            v19 = *(_QWORD *)(v15 + 8) + v16;
          }
          LOWORD(v22) = 1283;
          v26[1] = v23;
          v26[0] = &v21;
          v27 = 2;
          v28 = v24;
          sub_3909CF0(a1, v26, 0, 0, v13, v14, v19, v17, "invalid variant on expression '", &v20);
          result = a2;
        }
        else
        {
          result = sub_38CF310(*(_QWORD *)(a2 + 24), v3, *(_QWORD *)(a1 + 320), 0);
        }
        break;
      case 3:
        v11 = sub_38E57D0(a1, *(_QWORD *)(a2 + 24), v3);
        if ( v11 )
          result = sub_38CB340(*(_DWORD *)(a2 + 16), v11, *(_QWORD *)(a1 + 320), 0);
        else
LABEL_3:
          result = 0;
        break;
    }
  }
  return result;
}
