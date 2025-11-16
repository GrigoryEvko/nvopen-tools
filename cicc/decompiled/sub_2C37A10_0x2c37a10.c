// Function: sub_2C37A10
// Address: 0x2c37a10
//
char __fastcall sub_2C37A10(__int64 *a1, __int64 a2, int a3, __int64 a4)
{
  __int64 *v5; // r12
  __int64 v6; // r13
  __int64 v7; // rax
  unsigned __int64 v8; // rbx
  __int64 v9; // r14
  __int64 *v10; // r15
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rcx
  __int64 v14; // rax
  char v15; // dl
  __int64 *v16; // rax
  __int64 v17; // rax
  _QWORD *v18; // r13
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // r8
  __int64 v22; // r9
  __int64 v23; // rax
  __int64 v24; // rbx
  __int64 v25; // r15
  __int64 v26; // r12
  __int64 v27; // rsi
  __int64 v28; // rdx
  __int64 v29; // rcx
  __int64 v30; // r8
  __int64 v31; // r9
  __int64 v32; // rdx
  __int64 v33; // rcx
  __int64 v34; // r8
  __int64 v35; // r9
  __int64 v36; // r8
  __int64 v37; // r9
  __int64 v38; // rdx
  __int64 v39; // rcx
  __int64 v40; // rsi
  char v41; // al
  __int64 v42; // rdx
  __int64 v43; // rcx
  __int64 v44; // r8
  __int64 v45; // r9
  __int64 v46; // rdx
  __int64 v47; // rcx
  __int64 v48; // r8
  __int64 v49; // r9
  __int64 v50; // rdx
  __int64 v51; // rcx
  __int64 v52; // r8
  __int64 v53; // r9
  __int64 v54; // rdx
  __int64 v55; // rcx
  __int64 v56; // r8
  __int64 v57; // r9
  _QWORD *v59; // [rsp+10h] [rbp-260h]
  __int64 v60; // [rsp+18h] [rbp-258h]
  _BYTE *v61; // [rsp+20h] [rbp-250h]
  unsigned __int64 v62; // [rsp+20h] [rbp-250h]
  __int64 v63; // [rsp+20h] [rbp-250h]
  _BYTE *v64; // [rsp+28h] [rbp-248h]
  __int64 v65; // [rsp+28h] [rbp-248h]
  __int64 v68; // [rsp+40h] [rbp-230h] BYREF
  __int64 v69; // [rsp+48h] [rbp-228h]
  __int64 v70[12]; // [rsp+50h] [rbp-220h] BYREF
  __int64 v71; // [rsp+B0h] [rbp-1C0h]
  __int64 v72; // [rsp+B8h] [rbp-1B8h]
  __int64 v73[12]; // [rsp+D0h] [rbp-1A0h] BYREF
  __int64 v74; // [rsp+130h] [rbp-140h]
  __int64 v75; // [rsp+138h] [rbp-138h]
  void *v76[4]; // [rsp+150h] [rbp-120h] BYREF
  __int16 v77; // [rsp+170h] [rbp-100h]
  _BYTE v78[168]; // [rsp+1C8h] [rbp-A8h] BYREF

  v5 = a1;
  v6 = sub_2BF3F10(a1);
  v7 = sub_2BF0520(v6);
  v8 = *(_QWORD *)(v7 + 112) & 0xFFFFFFFFFFFFFFF8LL;
  if ( !v8 )
LABEL_49:
    BUG();
  if ( *(_BYTE *)(v8 - 16) == 4 )
  {
    v9 = v7;
    LOBYTE(v7) = *(_BYTE *)(v8 + 136);
    if ( (_BYTE)v7 == 78
      || (_BYTE)v7 == 79
      && (v7 = sub_2BF04A0(**(_QWORD **)(v8 + 24))) != 0
      && *(_BYTE *)(v7 + 8) == 4
      && *(_BYTE *)(v7 + 160) == 70
      && (v7 = sub_2BF04A0(**(_QWORD **)(v7 + 48))) != 0
      && *(_BYTE *)(v7 + 8) == 4
      && *(_BYTE *)(v7 + 160) == 73 )
    {
      v10 = *(__int64 **)(a4 + 112);
      v64 = (_BYTE *)sub_2C46ED0(a1[25], v10);
      LODWORD(v69) = a2 * a3;
      BYTE4(v69) = BYTE4(a2);
      v11 = sub_D95540((__int64)v64);
      v61 = sub_DCAEB0(v10, v11, v69);
      LOBYTE(v7) = sub_D968A0((__int64)v64);
      if ( !(_BYTE)v7 )
      {
        LOBYTE(v7) = sub_DC3A60((__int64)v10, 37, v64, v61);
        if ( (_BYTE)v7 )
        {
          v65 = *(_QWORD *)(v6 + 112);
          v12 = sub_2AAFF80((__int64)a1);
          if ( !*(_DWORD *)(v12 + 56) )
            BUG();
          v59 = *(_QWORD **)(*(_QWORD *)(**(_QWORD **)(v12 + 48) + 40LL) + 8LL);
          v13 = sub_2BF05A0(v65);
          v14 = *(_QWORD *)(v65 + 120);
          if ( v14 == v13 )
          {
LABEL_28:
            v23 = sub_2BF05A0(v65);
            if ( *(_QWORD *)(v65 + 120) != v23 )
            {
              v62 = v8;
              v24 = *(_QWORD *)(v65 + 120);
              v25 = v23;
              do
              {
                v26 = v24;
                v27 = 0;
                v24 = *(_QWORD *)(v24 + 8);
                if ( *(_DWORD *)(v26 + 32) )
                  v27 = **(_QWORD **)(v26 + 24);
                sub_2BF1250(v26 + 72, v27);
                sub_2C19E60((__int64 *)(v26 - 24));
              }
              while ( v24 != v25 );
              v8 = v62;
              v5 = a1;
            }
            v60 = 0;
            if ( *(_DWORD *)(v6 + 64) == 1 )
              v60 = **(_QWORD **)(v6 + 56);
            v63 = 0;
            if ( *(_DWORD *)(v6 + 88) == 1 )
              v63 = **(_QWORD **)(v6 + 80);
            sub_2C29140(v60, v6);
            sub_2C29140(v6, v63);
            sub_2C2F730(v76, *(_QWORD *)(v6 + 112));
            sub_2ABD910(v70, (__int64)v76, v28, v29, v30, v31);
            sub_2ABD910(v73, (__int64)v78, v32, v33, v34, v35);
            while ( 1 )
            {
              v38 = v72;
              v39 = v71;
              v40 = v74;
              if ( v72 - v71 == v75 - v74 )
              {
                if ( v71 == v72 )
                {
LABEL_47:
                  sub_2AB1B10((__int64)v73);
                  sub_2AB1B10((__int64)v70);
                  sub_2AB1B10((__int64)v78);
                  sub_2AB1B10((__int64)v76);
                  sub_2AB9570(v60 + 80, v65, v42, v43, v44, v45);
                  sub_2AB9570(v65 + 56, v60, v46, v47, v48, v49);
                  sub_2AB9570(v9 + 80, v63, v50, v51, v52, v53);
                  sub_2AB9570(v63 + 56, v9, v54, v55, v56, v57);
                  sub_2C36780(v5, v59);
                  goto LABEL_26;
                }
                while ( *(_QWORD *)v39 == *(_QWORD *)v40 )
                {
                  v41 = *(_BYTE *)(v39 + 16);
                  if ( v41 != *(_BYTE *)(v40 + 16) || v41 && *(_QWORD *)(v39 + 8) != *(_QWORD *)(v40 + 8) )
                    break;
                  v39 += 24;
                  v40 += 24;
                  if ( v72 == v39 )
                    goto LABEL_47;
                }
              }
              *(_QWORD *)(*(_QWORD *)(v72 - 24) + 48LL) = 0;
              sub_2ADA290((__int64)v70, v40, v38, v39, v36, v37);
            }
          }
          while ( 1 )
          {
            if ( !v14 )
              goto LABEL_49;
            v15 = *(_BYTE *)(v14 - 16);
            if ( v15 != 29 && v15 != 32 )
              break;
            v14 = *(_QWORD *)(v14 + 8);
            if ( v13 == v14 )
              goto LABEL_28;
          }
          v16 = (__int64 *)sub_B2BE50(*v10);
          v17 = sub_ACD6D0(v16);
          v70[0] = sub_2AC42A0((__int64)a1, v17);
          v68 = *(_QWORD *)(v8 + 64);
          if ( v68 )
            sub_2C25AB0(&v68);
          v77 = 257;
          v18 = (_QWORD *)sub_22077B0(0xC8u);
          if ( v18 )
          {
            v73[0] = v68;
            if ( v68 )
              sub_2C25AB0(v73);
            sub_2C26D30((__int64)v18, 79, v70, 1, v73, v76);
            sub_9C6650(v73);
          }
          sub_9C6650(&v68);
          sub_2AAFF40(v9, v18, (unsigned __int64 *)(v9 + 112));
LABEL_26:
          sub_2C19E60((__int64 *)(v8 - 24));
          sub_2C35B20((__int64)v5, a2, v19, v20, v21, v22);
          LOBYTE(v7) = sub_2C35DD0((__int64)v5, a3);
        }
      }
    }
  }
  return v7;
}
