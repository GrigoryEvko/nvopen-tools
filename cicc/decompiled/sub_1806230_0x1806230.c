// Function: sub_1806230
// Address: 0x1806230
//
unsigned __int64 __fastcall sub_1806230(__int64 *a1, __int64 a2, __int64 ***a3)
{
  unsigned __int64 result; // rax
  __int64 v4; // rdx
  unsigned __int64 v5; // r12
  __int64 v6; // r13
  unsigned int v7; // r14d
  __int64 v8; // r13
  __int64 v9; // rdx
  __int64 v10; // rbx
  __int64 v11; // rdx
  __int64 v12; // rdi
  __int64 v13; // rax
  __int64 v14; // r13
  __int64 v15; // r14
  __int64 v16; // rax
  unsigned __int8 *v17; // rsi
  __int64 v18; // rbx
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // r13
  __int64 v22; // rdi
  __int64 v23; // r15
  _QWORD *v24; // r12
  unsigned __int64 v25; // [rsp+8h] [rbp-D8h]
  unsigned __int64 v27; // [rsp+20h] [rbp-C0h]
  __int64 v29; // [rsp+38h] [rbp-A8h] BYREF
  unsigned __int8 *v30[2]; // [rsp+40h] [rbp-A0h] BYREF
  __int16 v31; // [rsp+50h] [rbp-90h]
  __int64 v32[5]; // [rsp+60h] [rbp-80h] BYREF
  int v33; // [rsp+88h] [rbp-58h]
  __int64 v34; // [rsp+90h] [rbp-50h]
  __int64 v35; // [rsp+98h] [rbp-48h]

  result = sub_16321C0(a2, (__int64)"llvm.global_ctors", 17, 0);
  if ( result )
  {
    result = *(_QWORD *)(result - 24);
    if ( *(_BYTE *)(result + 16) == 6 )
    {
      v4 = 24LL * (*(_DWORD *)(result + 20) & 0xFFFFFFF);
      if ( (*(_BYTE *)(result + 23) & 0x40) != 0 )
      {
        v5 = *(_QWORD *)(result - 8);
        v27 = v5 + v4;
      }
      else
      {
        v27 = result;
        v5 = result - v4;
      }
      result = (unsigned __int64)&v29;
      if ( v5 != v27 )
      {
        while ( 1 )
        {
          v8 = *(_QWORD *)v5;
          if ( *(_BYTE *)(*(_QWORD *)v5 + 16LL) == 10 )
            goto LABEL_11;
          v9 = 1LL - (*(_DWORD *)(v8 + 20) & 0xFFFFFFF);
          result = 3 * v9;
          v10 = *(_QWORD *)(v8 + 24 * v9);
          if ( *(_BYTE *)(v10 + 16) )
            goto LABEL_11;
          result = (unsigned __int64)sub_1649960(*(_QWORD *)(v8 + 24 * v9));
          if ( v11 != 16 || *(_QWORD *)result ^ 0x646F6D2E6E617361LL | *(_QWORD *)(result + 8) ^ 0x726F74635F656C75LL )
          {
            v6 = *(_QWORD *)(v8 - 24LL * (*(_DWORD *)(v8 + 20) & 0xFFFFFFF));
            if ( *(_BYTE *)(v6 + 16) != 13 )
              BUG();
            v7 = *(_DWORD *)(v6 + 32);
            if ( v7 > 0x40 )
            {
              if ( v7 - (unsigned int)sub_16A57B0(v6 + 24) > 0x40 )
              {
LABEL_18:
                v12 = *(_QWORD *)(v10 + 80);
                if ( v12 )
                  v12 -= 24;
                v13 = sub_157EE30(v12);
                v14 = *(_QWORD *)(v10 + 80);
                v15 = v13;
                if ( v14 )
                  v14 -= 24;
                v16 = sub_157E9C0(v14);
                v32[1] = v14;
                v32[0] = 0;
                v32[3] = v16;
                v32[4] = 0;
                v33 = 0;
                v34 = 0;
                v35 = 0;
                v32[2] = v15;
                if ( v15 != v14 + 40 )
                {
                  if ( !v15 )
                    BUG();
                  v17 = *(unsigned __int8 **)(v15 + 24);
                  v30[0] = v17;
                  if ( v17 )
                  {
                    sub_1623A60((__int64)v30, (__int64)v17, 2);
                    if ( v32[0] )
                      sub_161E7C0((__int64)v32, v32[0]);
                    v32[0] = (__int64)v30[0];
                    if ( v30[0] )
                      sub_1623210((__int64)v30, v30[0], (__int64)v32);
                  }
                }
                v18 = v10 + 72;
                v19 = sub_15A4A70(a3, a1[26]);
                v31 = 257;
                v20 = a1[38];
                v29 = v19;
                result = sub_1285290(v32, *(_QWORD *)(v20 + 24), v20, (int)&v29, 1, (__int64)v30, 0);
                v21 = *(_QWORD *)(v18 + 8);
                if ( v21 != v18 )
                {
                  v25 = v5;
                  do
                  {
                    v22 = v21 - 24;
                    if ( !v21 )
                      v22 = 0;
                    result = sub_157EBA0(v22);
                    v23 = result;
                    if ( *(_BYTE *)(result + 16) == 25 )
                    {
                      v31 = 257;
                      v24 = (_QWORD *)a1[39];
                      result = (unsigned __int64)sub_1648A60(72, 1u);
                      if ( result )
                        result = sub_15F5ED0(result, v24, (__int64)v30, v23);
                    }
                    v21 = *(_QWORD *)(v21 + 8);
                  }
                  while ( v18 != v21 );
                  v5 = v25;
                }
                if ( v32[0] )
                  result = sub_161E7C0((__int64)v32, v32[0]);
                goto LABEL_11;
              }
              result = **(_QWORD **)(v6 + 24);
            }
            else
            {
              result = *(_QWORD *)(v6 + 24);
            }
            if ( result > 1 )
              goto LABEL_18;
LABEL_11:
            v5 += 24LL;
            if ( v27 == v5 )
              return result;
          }
          else
          {
            v5 += 24LL;
            if ( v27 == v5 )
              return result;
          }
        }
      }
    }
  }
  return result;
}
