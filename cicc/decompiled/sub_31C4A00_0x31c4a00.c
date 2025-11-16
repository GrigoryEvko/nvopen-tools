// Function: sub_31C4A00
// Address: 0x31c4a00
//
__int64 __fastcall sub_31C4A00(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  __int64 v5; // r15
  __int64 result; // rax
  __int64 v7; // rdi
  __int64 v8; // r13
  __int64 v9; // r14
  __int64 v10; // rdi
  __int64 v11; // rdi
  bool v12; // al
  __int64 *v13; // rdi
  char v14; // al
  __int64 *v15; // rax
  char v16; // al
  __int64 *v17; // rdi
  char v18; // al
  __int64 *v19; // rax
  char v20; // al
  unsigned __int8 *v21; // [rsp+10h] [rbp-60h]
  unsigned __int8 *v22; // [rsp+10h] [rbp-60h]
  __int64 v23; // [rsp+18h] [rbp-58h]
  __int64 v24; // [rsp+20h] [rbp-50h] BYREF
  __int64 v25; // [rsp+28h] [rbp-48h]
  __int64 (__fastcall *v26)(__int64 *, __int64 *, int); // [rsp+30h] [rbp-40h]
  char (__fastcall *v27)(__int64 *, __int64 *); // [rsp+38h] [rbp-38h]

  *(_QWORD *)(a1 + 32) = a1 + 48;
  *(_QWORD *)(a1 + 80) = a3;
  *(_QWORD *)(a1 + 168) = a3;
  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 88) = 0;
  *(_QWORD *)(a1 + 120) = a1 + 136;
  *(_QWORD *)(a1 + 128) = 0;
  *(_QWORD *)(a1 + 136) = 0;
  v4 = *(_QWORD *)(a2 + 24);
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_DWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 64) = 0;
  *(_DWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 96) = 0;
  *(_QWORD *)(a1 + 104) = 0;
  *(_DWORD *)(a1 + 112) = 0;
  *(_QWORD *)(a1 + 144) = 0;
  *(_QWORD *)(a1 + 152) = 0;
  *(_DWORD *)(a1 + 160) = 0;
  *(_QWORD *)(a1 + 176) = v4;
  *(_QWORD *)(a1 + 184) = 0;
  v5 = sub_22416F0(&qword_5035C08, "stores", 0, 6u);
  v23 = sub_22416F0(&qword_5035C08, "loads", 0, 5u);
  result = v5 & v23;
  if ( (v5 & v23) != 0xFFFFFFFFFFFFFFFFLL )
  {
    v7 = *(_QWORD *)(a1 + 176);
    v24 = a1;
    v27 = sub_31C3110;
    v26 = sub_31C2710;
    *(_QWORD *)(a1 + 184) = sub_318ADE0(v7, (__int64)&v24);
    if ( v26 )
      v26(&v24, &v24, 3);
    result = sub_371B570(&v24, a2);
    v8 = *(_QWORD *)(a2 + 16) + 48LL;
    if ( v8 != v25 )
    {
      while ( 1 )
      {
        v9 = sub_371B3B0(&v24, v25, v26);
        if ( sub_318B670(v9) )
        {
          if ( v9 )
          {
            if ( v5 != -1 )
            {
              v21 = *(unsigned __int8 **)(v9 + 16);
              if ( !sub_B46500(v21) && (v21[2] & 1) == 0 )
              {
                if ( sub_318B630(v9) && (*(_DWORD *)(v9 + 8) != 37 || sub_318B6C0(v9)) )
                {
                  v11 = v9;
                  if ( sub_318B670(v9) )
                  {
                    v11 = sub_318B680(v9);
                  }
                  else if ( *(_DWORD *)(v9 + 8) == 37 )
                  {
                    v11 = sub_318B6C0(v9);
                  }
                }
                else
                {
                  v11 = v9;
                }
                v13 = sub_318EB80(v11);
                v14 = *(_BYTE *)(*v13 + 8);
                if ( (v14 & 0xFD) != 4 && v14 != 18 )
                {
                  if ( v14 == 17 )
                  {
                    v15 = sub_318E560(v13);
                    v16 = sub_318E520(v15);
                  }
                  else
                  {
                    v16 = sub_318E520(v13);
                  }
                  if ( v16 )
                    sub_31C4610(a1, v9);
                }
              }
            }
            if ( !sub_318B640(v9) )
              goto LABEL_14;
LABEL_11:
            if ( v23 != -1 )
            {
              v22 = *(unsigned __int8 **)(v9 + 16);
              if ( !sub_B46500(v22) && (v22[2] & 1) == 0 )
              {
                if ( sub_318B630(v9) && (*(_DWORD *)(v9 + 8) != 37 || sub_318B6C0(v9)) )
                {
                  v10 = v9;
                  if ( sub_318B670(v9) )
                  {
                    v10 = sub_318B680(v9);
                  }
                  else if ( *(_DWORD *)(v9 + 8) == 37 )
                  {
                    v10 = sub_318B6C0(v9);
                  }
                }
                else
                {
                  v10 = v9;
                }
                v17 = sub_318EB80(v10);
                v18 = *(_BYTE *)(*v17 + 8);
                if ( (v18 & 0xFD) != 4 && v18 != 18 )
                {
                  if ( v18 == 17 )
                  {
                    v19 = sub_318E560(v17);
                    v20 = sub_318E520(v19);
                  }
                  else
                  {
                    v20 = sub_318E520(v17);
                  }
                  if ( v20 )
                    sub_31C4220(a1 + 88, v9);
                }
              }
            }
            goto LABEL_14;
          }
          sub_318B640(0);
        }
        else
        {
          v12 = sub_318B640(v9);
          if ( v9 && v12 )
            goto LABEL_11;
        }
LABEL_14:
        result = (unsigned int)(*(_DWORD *)(a1 + 40) + *(_DWORD *)(a1 + 128));
        if ( (unsigned int)qword_5035B28 >= (unsigned int)result )
        {
          result = sub_371B2F0(&v24);
          if ( v8 != v25 )
            continue;
        }
        return result;
      }
    }
  }
  return result;
}
