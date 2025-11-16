// Function: sub_1AEA760
// Address: 0x1aea760
//
unsigned __int64 __fastcall sub_1AEA760(__int64 a1, __int64 a2, _QWORD *a3, int a4)
{
  unsigned __int64 result; // rax
  __int64 v6; // r12
  __int64 *v7; // rax
  __int64 v8; // rbx
  __int64 v9; // rdi
  __int64 v10; // r12
  __int64 v11; // rsi
  __int64 v12; // rdx
  __int64 v13; // r13
  __int64 v14; // r15
  _QWORD *v15; // rdx
  __int64 v16; // rax
  __int64 v17; // rsi
  __int64 v18; // rdx
  int v19; // r8d
  _BYTE *v20; // rax
  __int64 v21; // rdx
  _BYTE *v22; // r9
  size_t v23; // r11
  unsigned __int64 v24; // r15
  __int64 v25; // rsi
  size_t v26; // [rsp+0h] [rbp-A0h]
  _BYTE *v27; // [rsp+8h] [rbp-98h]
  _BYTE *v28; // [rsp+10h] [rbp-90h]
  _QWORD *v29; // [rsp+18h] [rbp-88h]
  __int64 v32; // [rsp+38h] [rbp-68h] BYREF
  _QWORD *v33; // [rsp+40h] [rbp-60h] BYREF
  __int64 v34; // [rsp+48h] [rbp-58h]
  _QWORD v35[10]; // [rsp+50h] [rbp-50h] BYREF

  result = sub_161E8E0(a1);
  if ( result )
  {
    v6 = result;
    v7 = (__int64 *)sub_16498A0(a1);
    result = sub_1629050(v7, v6);
    if ( result )
    {
      v8 = *(_QWORD *)(result + 8);
      if ( v8 )
      {
        v29 = a3;
        do
        {
          v9 = v8;
          v8 = *(_QWORD *)(v8 + 8);
          result = (unsigned __int64)sub_1648700(v9);
          v10 = result;
          if ( *(_BYTE *)(result + 16) != 78 )
            continue;
          result = *(_QWORD *)(result - 24);
          if ( *(_BYTE *)(result + 16) || (*(_BYTE *)(result + 33) & 0x20) == 0 || *(_DWORD *)(result + 36) != 38 )
            continue;
          v11 = *(_QWORD *)(v10 + 48);
          v32 = v11;
          if ( v11 )
          {
            sub_1623A60((__int64)&v32, v11, 2);
            v12 = *(_DWORD *)(v10 + 20) & 0xFFFFFFF;
            v13 = *(_QWORD *)(*(_QWORD *)(v10 + 24 * (1 - v12)) + 24LL);
            result = *(_QWORD *)(v10 + 24 * (2 - v12));
            v14 = *(_QWORD *)(result + 24);
            if ( !v14
              || (v15 = *(_QWORD **)(v14 + 24),
                  result = (__int64)(*(_QWORD *)(v14 + 32) - (_QWORD)v15) >> 3,
                  !(_DWORD)result) )
            {
LABEL_18:
              v17 = v32;
              if ( !v32 )
                continue;
              goto LABEL_17;
            }
          }
          else
          {
            v18 = *(_DWORD *)(v10 + 20) & 0xFFFFFFF;
            v13 = *(_QWORD *)(*(_QWORD *)(v10 + 24 * (1 - v18)) + 24LL);
            result = *(_QWORD *)(v10 + 24 * (2 - v18));
            v14 = *(_QWORD *)(result + 24);
            if ( !v14 )
              continue;
            v15 = *(_QWORD **)(v14 + 24);
            result = (__int64)(*(_QWORD *)(v14 + 32) - (_QWORD)v15) >> 3;
            if ( !(_DWORD)result )
              continue;
          }
          if ( *v15 != 6 )
            goto LABEL_18;
          if ( a4 )
          {
            v34 = 0x400000001LL;
            v33 = v35;
            v35[0] = 6;
            sub_15B13F0((__int64)&v33, a4);
            v20 = *(_BYTE **)(v14 + 32);
            v21 = (unsigned int)v34;
            v22 = (_BYTE *)(*(_QWORD *)(v14 + 24) + 8LL);
            v23 = v20 - v22;
            v24 = (v20 - v22) >> 3;
            if ( v24 > HIDWORD(v34) - (unsigned __int64)(unsigned int)v34 )
            {
              v26 = v20 - v22;
              v27 = v22;
              v28 = v20;
              sub_16CD150((__int64)&v33, v35, v24 + (unsigned int)v34, 8, v19, (int)v22);
              v21 = (unsigned int)v34;
              v23 = v26;
              v22 = v27;
              v20 = v28;
            }
            v25 = (__int64)v33;
            if ( v20 != v22 )
            {
              memcpy(&v33[v21], v22, v23);
              v25 = (__int64)v33;
              LODWORD(v21) = v34;
            }
            LODWORD(v34) = v24 + v21;
            v14 = sub_15A6870((__int64)v29, v25, (unsigned int)(v24 + v21));
            if ( v33 != v35 )
              _libc_free((unsigned __int64)v33);
          }
          v16 = sub_15C70A0((__int64)&v32);
          sub_15A76D0(v29, a2, v13, v14, v16, v10);
          result = sub_15F20C0((_QWORD *)v10);
          v17 = v32;
          if ( !v32 )
            continue;
LABEL_17:
          result = sub_161E7C0((__int64)&v32, v17);
        }
        while ( v8 );
      }
    }
  }
  return result;
}
