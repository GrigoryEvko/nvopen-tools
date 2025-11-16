// Function: sub_1C9A0C0
// Address: 0x1c9a0c0
//
__int64 __fastcall sub_1C9A0C0(__int64 a1, __int64 *a2, unsigned __int8 a3, __int64 a4)
{
  __int64 result; // rax
  __int64 *v6; // r12
  __int64 v7; // rax
  __int64 v8; // rdx
  unsigned __int64 v9; // rax
  __int64 v10; // rax
  _QWORD *v11; // rax
  __int64 v12; // rax
  __int64 v13; // rax
  _BYTE *v14; // rsi
  __int64 v15; // r13
  _QWORD *v16; // rax
  __int64 v17; // rax
  __int64 v18; // rax
  _BYTE *v19; // rsi
  __int64 *v20; // rcx
  __int64 *v21; // r15
  __int64 v22; // r14
  _QWORD *v23; // rax
  __int64 v24; // r13
  __int64 v25; // rax
  __int64 *v26; // rax
  __int64 *v27; // rax
  int v28; // r8d
  __int64 *v29; // r10
  __int64 *v30; // rax
  __int64 v31; // rdx
  __int64 *v32; // rax
  _QWORD *v33; // rdi
  __int64 *v34; // rax
  int v35; // [rsp+4h] [rbp-BCh]
  __int64 *v37; // [rsp+10h] [rbp-B0h]
  __int64 *v38; // [rsp+18h] [rbp-A8h]
  int v39; // [rsp+18h] [rbp-A8h]
  __int64 v40; // [rsp+20h] [rbp-A0h]
  __int64 v41; // [rsp+28h] [rbp-98h]
  unsigned int v44; // [rsp+4Ch] [rbp-74h] BYREF
  __int64 *v45; // [rsp+50h] [rbp-70h] BYREF
  _BYTE *v46; // [rsp+58h] [rbp-68h]
  _BYTE *v47; // [rsp+60h] [rbp-60h]
  _QWORD v48[2]; // [rsp+70h] [rbp-50h] BYREF
  char v49; // [rsp+80h] [rbp-40h]
  char v50; // [rsp+81h] [rbp-3Fh]

  result = *a2;
  v41 = *a2;
  if ( *(_BYTE *)(*a2 + 8) == 13 )
  {
    result = *(unsigned int *)(result + 12);
    v44 = 0;
    v35 = result;
    if ( (_DWORD)result )
    {
      do
      {
        v50 = 1;
        v48[0] = "extract";
        v49 = 3;
        v6 = sub_1648A60(88, 1u);
        if ( v6 )
        {
          v7 = sub_15FB2A0(*a2, &v44, 1);
          sub_15F1EA0((__int64)v6, v7, 62, (__int64)(v6 - 3), 1, a4);
          if ( *(v6 - 3) )
          {
            v8 = *(v6 - 2);
            v9 = *(v6 - 1) & 0xFFFFFFFFFFFFFFFCLL;
            *(_QWORD *)v9 = v8;
            if ( v8 )
              *(_QWORD *)(v8 + 16) = *(_QWORD *)(v8 + 16) & 3LL | v9;
          }
          *(v6 - 3) = (__int64)a2;
          v10 = a2[1];
          *(v6 - 2) = v10;
          if ( v10 )
            *(_QWORD *)(v10 + 16) = (unsigned __int64)(v6 - 2) | *(_QWORD *)(v10 + 16) & 3LL;
          *(v6 - 1) = (unsigned __int64)(a2 + 1) | *(v6 - 1) & 3;
          a2[1] = (__int64)(v6 - 3);
          v6[7] = (__int64)(v6 + 9);
          v6[8] = 0x400000000LL;
          sub_15FB110((__int64)v6, &v44, 1, (__int64)v48);
        }
        v45 = 0;
        v46 = 0;
        v47 = 0;
        v11 = (_QWORD *)sub_16498A0(a4);
        v12 = sub_1643350(v11);
        v13 = sub_159C470(v12, 0, 0);
        v14 = v46;
        v48[0] = v13;
        if ( v46 == v47 )
        {
          sub_12879C0((__int64)&v45, v46, v48);
        }
        else
        {
          if ( v46 )
          {
            *(_QWORD *)v46 = v13;
            v14 = v46;
          }
          v46 = v14 + 8;
        }
        v15 = v44;
        v16 = (_QWORD *)sub_16498A0(a4);
        v17 = sub_1643350(v16);
        v18 = sub_159C470(v17, v15, 0);
        v19 = v46;
        v48[0] = v18;
        if ( v46 == v47 )
        {
          sub_12879C0((__int64)&v45, v46, v48);
          v20 = (__int64 *)v46;
        }
        else
        {
          if ( v46 )
          {
            *(_QWORD *)v46 = v18;
            v19 = v46;
          }
          v20 = (__int64 *)(v19 + 8);
          v46 = v19 + 8;
        }
        v21 = v45;
        v50 = 1;
        v49 = 3;
        v48[0] = "gep";
        v22 = v20 - v45;
        v38 = v20;
        v23 = sub_1648A60(72, (int)v22 + 1);
        v24 = (__int64)v23;
        if ( v23 )
        {
          v40 = (__int64)&v23[-3 * (unsigned int)(v22 + 1)];
          v25 = *(_QWORD *)a1;
          if ( *(_BYTE *)(*(_QWORD *)a1 + 8LL) == 16 )
            v25 = **(_QWORD **)(v25 + 16);
          v37 = v38;
          v39 = *(_DWORD *)(v25 + 8) >> 8;
          v26 = (__int64 *)sub_15F9F50(v41, (__int64)v21, v22);
          v27 = (__int64 *)sub_1646BA0(v26, v39);
          v28 = v22 + 1;
          v29 = v27;
          if ( *(_BYTE *)(*(_QWORD *)a1 + 8LL) == 16 )
          {
            v34 = sub_16463B0(v27, *(_QWORD *)(*(_QWORD *)a1 + 32LL));
            v28 = v22 + 1;
            v29 = v34;
          }
          else if ( v21 != v37 )
          {
            v30 = v21;
            while ( 1 )
            {
              v31 = *(_QWORD *)*v30;
              if ( *(_BYTE *)(v31 + 8) == 16 )
                break;
              if ( v37 == ++v30 )
                goto LABEL_27;
            }
            v32 = sub_16463B0(v29, *(_QWORD *)(v31 + 32));
            v28 = v22 + 1;
            v29 = v32;
          }
LABEL_27:
          sub_15F1EA0(v24, (__int64)v29, 32, v40, v28, a4);
          *(_QWORD *)(v24 + 56) = v41;
          *(_QWORD *)(v24 + 64) = sub_15F9F50(v41, (__int64)v21, v22);
          sub_15F9CE0(v24, a1, v21, v22, (__int64)v48);
        }
        sub_15FA2E0(v24, 1);
        if ( *(_BYTE *)(*v6 + 8) == 13 && (unsigned __int8)sub_1C97B40(*v6) )
        {
          sub_1C9A0C0(v24, v6, a3, a4);
        }
        else
        {
          v33 = sub_1648A60(64, 2u);
          if ( v33 )
            sub_15F9650((__int64)v33, (__int64)v6, v24, a3, a4);
        }
        if ( v45 )
          j_j___libc_free_0(v45, v47 - (_BYTE *)v45);
        result = v44 + 1;
        v44 = result;
      }
      while ( (_DWORD)result != v35 );
    }
  }
  return result;
}
