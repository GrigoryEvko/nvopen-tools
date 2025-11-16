// Function: sub_30270A0
// Address: 0x30270a0
//
unsigned __int64 __fastcall sub_30270A0(__int64 a1, unsigned __int8 *a2, char a3)
{
  unsigned __int8 *v3; // r12
  unsigned __int8 v4; // bl
  _QWORD *v5; // r14
  int v6; // edx
  _QWORD *v7; // rsi
  _QWORD *v8; // rdi
  __int64 v10; // rax
  unsigned __int64 v11; // rax
  unsigned __int64 v12; // r12
  __int64 v13; // r14
  __int64 v14; // r15
  __int64 v15; // rax
  __int64 v16; // rax
  unsigned int v17; // eax
  __int64 v18; // r9
  __int64 v19; // rax
  int v20; // ebx
  __int64 v21; // rdi
  unsigned __int64 v22; // rax
  unsigned int v23; // r15d
  __int64 v24; // rbx
  __int64 v25; // rdx
  __int64 v26; // r15
  __int64 v27; // rdx
  __int64 v28; // rdx
  unsigned __int64 v29; // rax
  __int64 v30; // rdx
  char v31; // al
  unsigned __int64 v32; // rax
  __int64 v33; // rcx
  __int64 v34; // [rsp+0h] [rbp-D0h]
  __int64 v35; // [rsp+8h] [rbp-C8h]
  __int64 v36; // [rsp+8h] [rbp-C8h]
  _QWORD v37[2]; // [rsp+10h] [rbp-C0h] BYREF
  char v38; // [rsp+20h] [rbp-B0h] BYREF
  unsigned __int64 v39; // [rsp+30h] [rbp-A0h] BYREF
  __int64 v40; // [rsp+38h] [rbp-98h]
  __int16 v41; // [rsp+50h] [rbp-80h]
  __int64 v42; // [rsp+60h] [rbp-70h] BYREF
  __int64 v43; // [rsp+68h] [rbp-68h]
  __int64 v44; // [rsp+70h] [rbp-60h]
  __int64 v45; // [rsp+78h] [rbp-58h]
  __int64 v46; // [rsp+80h] [rbp-50h]
  __int64 v47; // [rsp+88h] [rbp-48h]
  _QWORD *v48; // [rsp+90h] [rbp-40h]

  while ( 2 )
  {
    v3 = a2;
    v4 = a3;
    v5 = *(_QWORD **)(a1 + 216);
    if ( sub_AC30F0((__int64)a2) || (v6 = *a2, (unsigned int)(v6 - 12) <= 1) )
    {
      v7 = v5;
      v8 = 0;
      return sub_E81A90((__int64)v8, v7, 0, 0);
    }
    if ( (_BYTE)v6 == 17 )
    {
      v8 = (_QWORD *)*((_QWORD *)a2 + 3);
      if ( *((_DWORD *)a2 + 8) > 0x40u )
        v8 = (_QWORD *)*v8;
      v7 = v5;
      return sub_E81A90((__int64)v8, v7, 0, 0);
    }
    if ( (unsigned __int8)v6 > 3u )
    {
      if ( (_BYTE)v6 != 5 )
LABEL_45:
        BUG();
      switch ( *((_WORD *)a2 + 1) )
      {
        case 0xD:
          v23 = v4;
          v24 = sub_30270A0(a1, *(_QWORD *)&a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)], v4);
          v25 = sub_30270A0(a1, *(_QWORD *)&a2[32 * (1LL - (*((_DWORD *)a2 + 1) & 0x7FFFFFF))], v23);
          if ( *((_WORD *)a2 + 1) != 13 )
            goto LABEL_45;
          return sub_E81A00(0, v24, v25, v5, 0);
        case 0x22:
          v35 = sub_31DA930(a1);
          v17 = sub_AE43A0(v35, *((_QWORD *)a2 + 1));
          v18 = v35;
          LODWORD(v43) = v17;
          if ( v17 > 0x40 )
          {
            sub_C43690((__int64)&v42, 0, 0);
            v18 = v35;
          }
          else
          {
            v42 = 0;
          }
          sub_BB6360((__int64)a2, v18, (__int64)&v42, 0, 0);
          v19 = sub_30270A0(a1, *(_QWORD *)&a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)], v4);
          v20 = v43;
          v12 = v19;
          if ( (unsigned int)v43 <= 0x40 )
          {
            if ( !v42 )
              return v12;
            v21 = 0;
            if ( (_DWORD)v43 )
              v21 = v42 << (64 - (unsigned __int8)v43) >> (64 - (unsigned __int8)v43);
          }
          else
          {
            if ( v20 == (unsigned int)sub_C444A0((__int64)&v42) )
              goto LABEL_30;
            v21 = *(_QWORD *)v42;
          }
          v22 = sub_E81A90(v21, v5, 0, 0);
          v12 = sub_E81A00(0, v12, v22, v5, 0);
          if ( (unsigned int)v43 <= 0x40 )
            return v12;
LABEL_30:
          if ( v42 )
            j_j___libc_free_0_0(v42);
          break;
        case 0x26:
        case 0x31:
          a3 = v4;
          a2 = *(unsigned __int8 **)&a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
          continue;
        case 0x2F:
          v26 = sub_31DA930(a1);
          v34 = *((_QWORD *)a2 + 1);
          v36 = *(_QWORD *)&a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
          v12 = sub_30270A0(a1, v36, v4);
          v42 = sub_BDB740(v26, *(_QWORD *)(v36 + 8));
          v43 = v27;
          v39 = sub_BDB740(v26, v34);
          v40 = v28;
          if ( v39 == v42 && (_BYTE)v40 == (_BYTE)v43 )
            return v12;
          v29 = sub_BDB740(v26, *(_QWORD *)(v36 + 8));
          v43 = v30;
          v42 = 8 * v29;
          v31 = sub_CA1930(&v42);
          v32 = sub_E81A90(0xFFFFFFFFFFFFFFFFLL >> (64 - v31), v5, 0, 0);
          return sub_E81A00(1, v12, v32, v5, 0);
        case 0x30:
          v13 = sub_31DA930(a1);
          v14 = *(_QWORD *)&a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
          v15 = sub_AE4450(v13, *((_QWORD *)a2 + 1));
          a2 = (unsigned __int8 *)sub_96F3F0(v14, v15, 0, v13);
          if ( !a2 )
            goto LABEL_18;
          goto LABEL_19;
        case 0x32:
          if ( *(_DWORD *)(*((_QWORD *)a2 + 1) + 8LL) >> 8 )
            goto LABEL_18;
          a3 = 1;
          a2 = *(unsigned __int8 **)&a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
          continue;
        default:
LABEL_18:
          v16 = sub_31DA930(a1);
          a2 = (unsigned __int8 *)sub_97B670(v3, v16, 0);
          if ( a2 == v3 )
          {
            v37[0] = &v38;
            v47 = 0x100000000LL;
            v48 = v37;
            v42 = (__int64)&unk_49DD210;
            v37[1] = 0;
            v38 = 0;
            v43 = 0;
            v44 = 0;
            v45 = 0;
            v46 = 0;
            sub_CB5980((__int64)&v42, 0, 0, 0);
            sub_904010((__int64)&v42, "Unsupported expression in static initializer: ");
            v33 = *(_QWORD *)(a1 + 232);
            if ( v33 )
              v33 = *(_QWORD *)(*(_QWORD *)v33 + 40LL);
            sub_A5BF40(v3, (__int64)&v42, 0, v33);
            v41 = 260;
            v39 = (unsigned __int64)v48;
            sub_C64D30((__int64)&v39, 1u);
          }
LABEL_19:
          a3 = v4;
          continue;
      }
    }
    else
    {
      v10 = sub_31DB510(a1, a2);
      v11 = sub_E808D0(v10, 0, v5, 0);
      v12 = v11;
      if ( v4 )
      {
        v12 = sub_3058830(v11, v5);
        if ( v12 )
          v12 += 8LL;
      }
    }
    return v12;
  }
}
