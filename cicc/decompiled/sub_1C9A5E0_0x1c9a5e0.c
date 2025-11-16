// Function: sub_1C9A5E0
// Address: 0x1c9a5e0
//
__int64 __fastcall sub_1C9A5E0(__int64 a1, __int64 a2, __int64 a3, unsigned __int8 a4, __int64 a5)
{
  __int64 v5; // r12
  __int64 v8; // rax
  int v9; // ecx
  _QWORD *v10; // rax
  __int64 v11; // rax
  __int64 v12; // rax
  _BYTE *v13; // rsi
  __int64 v14; // r12
  _QWORD *v15; // rax
  __int64 v16; // rax
  __int64 v17; // rax
  _BYTE *v18; // rsi
  __int64 *v19; // rcx
  __int64 *v20; // r14
  __int64 v21; // r13
  _QWORD *v22; // rax
  __int64 v23; // r12
  __int64 v24; // rax
  __int64 *v25; // rax
  __int64 *v26; // rax
  int v27; // r8d
  __int64 *v28; // r10
  __int64 *v29; // rax
  __int64 v30; // rdx
  __int64 *v31; // rax
  __int64 v32; // r14
  __int64 *v33; // rax
  __int64 *v34; // r13
  _QWORD *v35; // rax
  __int64 *v37; // rax
  _BYTE *v38; // rsi
  int v40; // [rsp+Ch] [rbp-B4h]
  __int64 *v42; // [rsp+18h] [rbp-A8h]
  __int64 *v43; // [rsp+20h] [rbp-A0h]
  int v44; // [rsp+20h] [rbp-A0h]
  __int64 v45; // [rsp+28h] [rbp-98h]
  __int64 *v46; // [rsp+30h] [rbp-90h]
  unsigned int v47; // [rsp+4Ch] [rbp-74h] BYREF
  __int64 *v48; // [rsp+50h] [rbp-70h] BYREF
  _BYTE *v49; // [rsp+58h] [rbp-68h]
  _BYTE *v50; // [rsp+60h] [rbp-60h]
  _QWORD v51[2]; // [rsp+70h] [rbp-50h] BYREF
  char v52; // [rsp+80h] [rbp-40h]
  char v53; // [rsp+81h] [rbp-3Fh]

  v5 = 0;
  if ( *(_BYTE *)(a2 + 8) == 13 )
  {
    v8 = sub_1599EF0((__int64 **)a2);
    v9 = *(_DWORD *)(a2 + 12);
    v47 = 0;
    v46 = (__int64 *)v8;
    v5 = v8;
    v40 = v9;
    if ( v9 )
    {
      while ( 1 )
      {
        v48 = 0;
        v49 = 0;
        v50 = 0;
        v10 = (_QWORD *)sub_16498A0(a3);
        v11 = sub_1643350(v10);
        v12 = sub_159C470(v11, 0, 0);
        v13 = v49;
        v51[0] = v12;
        if ( v49 == v50 )
        {
          sub_12879C0((__int64)&v48, v49, v51);
        }
        else
        {
          if ( v49 )
          {
            *(_QWORD *)v49 = v12;
            v13 = v49;
          }
          v49 = v13 + 8;
        }
        v14 = v47;
        v15 = (_QWORD *)sub_16498A0(a3);
        v16 = sub_1643350(v15);
        v17 = sub_159C470(v16, v14, 0);
        v18 = v49;
        v51[0] = v17;
        if ( v49 == v50 )
        {
          sub_12879C0((__int64)&v48, v49, v51);
          v19 = (__int64 *)v49;
        }
        else
        {
          if ( v49 )
          {
            *(_QWORD *)v49 = v17;
            v18 = v49;
          }
          v19 = (__int64 *)(v18 + 8);
          v49 = v18 + 8;
        }
        v20 = v48;
        v53 = 1;
        v52 = 3;
        v51[0] = "gep";
        v21 = v19 - v48;
        v43 = v19;
        v22 = sub_1648A60(72, (int)v21 + 1);
        v23 = (__int64)v22;
        if ( v22 )
        {
          v45 = (__int64)&v22[-3 * (unsigned int)(v21 + 1)];
          v24 = *(_QWORD *)a1;
          if ( *(_BYTE *)(*(_QWORD *)a1 + 8LL) == 16 )
            v24 = **(_QWORD **)(v24 + 16);
          v42 = v43;
          v44 = *(_DWORD *)(v24 + 8) >> 8;
          v25 = (__int64 *)sub_15F9F50(a2, (__int64)v20, v21);
          v26 = (__int64 *)sub_1646BA0(v25, v44);
          v27 = v21 + 1;
          v28 = v26;
          if ( *(_BYTE *)(*(_QWORD *)a1 + 8LL) == 16 )
          {
            v37 = sub_16463B0(v26, *(_QWORD *)(*(_QWORD *)a1 + 32LL));
            v27 = v21 + 1;
            v28 = v37;
          }
          else if ( v20 != v42 )
          {
            v29 = v20;
            while ( 1 )
            {
              v30 = *(_QWORD *)*v29;
              if ( *(_BYTE *)(v30 + 8) == 16 )
                break;
              if ( v42 == ++v29 )
                goto LABEL_20;
            }
            v31 = sub_16463B0(v28, *(_QWORD *)(v30 + 32));
            v27 = v21 + 1;
            v28 = v31;
          }
LABEL_20:
          sub_15F1EA0(v23, (__int64)v28, 32, v45, v27, a3);
          *(_QWORD *)(v23 + 56) = a2;
          *(_QWORD *)(v23 + 64) = sub_15F9F50(a2, (__int64)v20, v21);
          sub_15F9CE0(v23, a1, v20, v21, (__int64)v51);
        }
        sub_15FA2E0(v23, 1);
        v32 = *(_QWORD *)(v23 + 64);
        v33 = sub_1648A60(64, 1u);
        v34 = v33;
        if ( v33 )
          sub_15F9210((__int64)v33, v32, v23, "loadfield", a4, a3);
        if ( *(_BYTE *)(*v34 + 8) == 13 && (unsigned __int8)sub_1C97B40(*v34) )
        {
          v51[0] = v34;
          v38 = *(_BYTE **)(a5 + 8);
          if ( v38 == *(_BYTE **)(a5 + 16) )
          {
            sub_17C2330(a5, v38, v51);
          }
          else
          {
            if ( v38 )
            {
              *(_QWORD *)v38 = v34;
              v38 = *(_BYTE **)(a5 + 8);
            }
            *(_QWORD *)(a5 + 8) = v38 + 8;
          }
          v34 = (__int64 *)sub_1C9A5E0(v23, *v34, a3, a4, a5);
        }
        v53 = 1;
        v51[0] = "insertfield";
        v52 = 3;
        v35 = sub_1648A60(88, 2u);
        v5 = (__int64)v35;
        if ( v35 )
        {
          sub_15F1EA0((__int64)v35, *v46, 63, (__int64)(v35 - 6), 2, a3);
          *(_QWORD *)(v5 + 56) = v5 + 72;
          *(_QWORD *)(v5 + 64) = 0x400000000LL;
          sub_15FAD90(v5, (__int64)v46, (__int64)v34, &v47, 1, (__int64)v51);
        }
        if ( v48 )
          j_j___libc_free_0(v48, v50 - (_BYTE *)v48);
        if ( v40 == ++v47 )
          break;
        v46 = (__int64 *)v5;
      }
    }
  }
  return v5;
}
