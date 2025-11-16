// Function: sub_2B21610
// Address: 0x2b21610
//
__int64 __fastcall sub_2B21610(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, char a6)
{
  __int64 v6; // r12
  __int64 v11; // r12
  __int64 v12; // rdx
  int v13; // r14d
  __int64 v14; // rbx
  __int64 v15; // r13
  __int64 v16; // rdx
  unsigned int v17; // esi
  unsigned int v18; // eax
  _QWORD *v20; // rdi
  int v21; // edx
  __int64 *v22; // rax
  __int64 v23; // rax
  __int64 v24; // rdx
  int v25; // r14d
  __int64 v26; // rbx
  __int64 v27; // r13
  __int64 v28; // rdx
  unsigned int v29; // esi
  _QWORD *v30; // rdi
  int v31; // edx
  __int64 *v32; // rax
  __int64 v33; // rax
  int v34; // eax
  __int64 v35; // r12
  __int64 v36; // r10
  __int64 v37; // r8
  __int64 v38; // rcx
  __int64 v39; // rdx
  __int64 v40; // rsi
  _QWORD *v41; // rax
  _QWORD *v42; // r10
  _QWORD **v43; // rdx
  int v44; // ecx
  __int64 *v45; // rax
  __int64 v46; // rax
  __int64 v47; // r12
  __int64 v48; // rdx
  unsigned int v49; // esi
  _QWORD *v50; // rdi
  int v51; // edx
  int v52; // eax
  __int64 *v53; // rax
  __int64 v54; // rdi
  __int64 v55; // rax
  _QWORD *v56; // rdi
  int v57; // edx
  __int64 *v58; // rax
  __int64 v59; // rdi
  __int64 v60; // rax
  __int64 v61; // [rsp+0h] [rbp-90h]
  unsigned int v62; // [rsp+8h] [rbp-88h]
  unsigned int v63; // [rsp+8h] [rbp-88h]
  __int64 v64; // [rsp+8h] [rbp-88h]
  __int64 v65; // [rsp+8h] [rbp-88h]
  __int64 v66; // [rsp+8h] [rbp-88h]
  __int64 v67; // [rsp+10h] [rbp-80h]
  __int64 v68; // [rsp+18h] [rbp-78h]
  __int64 v69; // [rsp+20h] [rbp-70h]
  __int64 v70; // [rsp+28h] [rbp-68h]
  __int64 v71[4]; // [rsp+30h] [rbp-60h] BYREF
  __int16 v72; // [rsp+50h] [rbp-40h]

  v6 = *(_QWORD *)(a3 + 8);
  switch ( (int)a2 )
  {
    case 1:
    case 2:
    case 5:
    case 10:
    case 11:
      v62 = sub_1022EF0(a2);
      v11 = (*(__int64 (__fastcall **)(_QWORD, _QWORD, __int64, __int64))(**(_QWORD **)(a1 + 80) + 16LL))(
              *(_QWORD *)(a1 + 80),
              v62,
              a3,
              a4);
      if ( !v11 )
      {
        v72 = 257;
        v11 = sub_B504D0(v62, a3, a4, (__int64)v71, 0, 0);
        if ( (unsigned __int8)sub_920620(v11) )
        {
          v12 = *(_QWORD *)(a1 + 96);
          v13 = *(_DWORD *)(a1 + 104);
          if ( v12 )
            sub_B99FD0(v11, 3u, v12);
          sub_B45150(v11, v13);
        }
        (*(void (__fastcall **)(_QWORD, __int64, __int64, _QWORD, _QWORD))(**(_QWORD **)(a1 + 88) + 16LL))(
          *(_QWORD *)(a1 + 88),
          v11,
          a5,
          *(_QWORD *)(a1 + 56),
          *(_QWORD *)(a1 + 64));
        v14 = *(_QWORD *)a1;
        v15 = *(_QWORD *)a1 + 16LL * *(unsigned int *)(a1 + 8);
        if ( *(_QWORD *)a1 != v15 )
        {
          do
          {
            v16 = *(_QWORD *)(v14 + 8);
            v17 = *(_DWORD *)v14;
            v14 += 16;
            sub_B99FD0(v11, v17, v16);
          }
          while ( v15 != v14 );
        }
      }
      return v11;
    case 3:
      if ( !a6
        || ((v30 = *(_QWORD **)v6, v31 = *(unsigned __int8 *)(v6 + 8), (unsigned int)(v31 - 17) > 1)
          ? (v33 = sub_BCB2A0(v30))
          : (BYTE4(v67) = (_BYTE)v31 == 18,
             LODWORD(v67) = *(_DWORD *)(v6 + 32),
             v32 = (__int64 *)sub_BCB2A0(v30),
             v33 = sub_BCE1B0(v32, v67)),
            v6 != v33) )
      {
        v34 = sub_1022EF0(3);
        return sub_2B21080((__int64 *)a1, v34, a3, a4, v71[0], 0, a5, 0);
      }
      v56 = *(_QWORD **)v6;
      v57 = *(unsigned __int8 *)(v6 + 8);
      if ( (unsigned int)(v57 - 17) > 1 )
      {
        v59 = sub_BCB2A0(v56);
      }
      else
      {
        BYTE4(v68) = (_BYTE)v57 == 18;
        LODWORD(v68) = *(_DWORD *)(v6 + 32);
        v58 = (__int64 *)sub_BCB2A0(v56);
        v59 = sub_BCE1B0(v58, v68);
      }
      v60 = sub_AD62B0(v59);
      v37 = a5;
      v38 = a4;
      v39 = v60;
      v40 = a3;
      return sub_B36550((unsigned int **)a1, v40, v39, v38, v37, 0);
    case 4:
      if ( a6
        && ((v20 = *(_QWORD **)v6, v21 = *(unsigned __int8 *)(v6 + 8), (unsigned int)(v21 - 17) > 1)
          ? (v23 = sub_BCB2A0(v20))
          : (BYTE4(v69) = (_BYTE)v21 == 18,
             LODWORD(v69) = *(_DWORD *)(v6 + 32),
             v22 = (__int64 *)sub_BCB2A0(v20),
             a2 = v69,
             v23 = sub_BCE1B0(v22, v69)),
            v6 == v23) )
      {
        v50 = *(_QWORD **)v6;
        v51 = *(unsigned __int8 *)(v6 + 8);
        if ( (unsigned int)(v51 - 17) > 1 )
        {
          v54 = sub_BCB2A0(v50);
        }
        else
        {
          v52 = *(_DWORD *)(v6 + 32);
          BYTE4(v71[0]) = (_BYTE)v51 == 18;
          LODWORD(v71[0]) = v52;
          v53 = (__int64 *)sub_BCB2A0(v50);
          a2 = v71[0];
          v54 = sub_BCE1B0(v53, v71[0]);
        }
        v55 = sub_AD6530(v54, a2);
        v37 = a5;
        v39 = a4;
        v38 = v55;
        v40 = a3;
        return sub_B36550((unsigned int **)a1, v40, v39, v38, v37, 0);
      }
      else
      {
        v63 = sub_1022EF0(4);
        v11 = (*(__int64 (__fastcall **)(_QWORD, _QWORD, __int64, __int64))(**(_QWORD **)(a1 + 80) + 16LL))(
                *(_QWORD *)(a1 + 80),
                v63,
                a3,
                a4);
        if ( !v11 )
        {
          v72 = 257;
          v11 = sub_B504D0(v63, a3, a4, (__int64)v71, 0, 0);
          if ( (unsigned __int8)sub_920620(v11) )
          {
            v24 = *(_QWORD *)(a1 + 96);
            v25 = *(_DWORD *)(a1 + 104);
            if ( v24 )
              sub_B99FD0(v11, 3u, v24);
            sub_B45150(v11, v25);
          }
          (*(void (__fastcall **)(_QWORD, __int64, __int64, _QWORD, _QWORD))(**(_QWORD **)(a1 + 88) + 16LL))(
            *(_QWORD *)(a1 + 88),
            v11,
            a5,
            *(_QWORD *)(a1 + 56),
            *(_QWORD *)(a1 + 64));
          v26 = *(_QWORD *)a1;
          v27 = *(_QWORD *)a1 + 16LL * *(unsigned int *)(a1 + 8);
          while ( v27 != v26 )
          {
            v28 = *(_QWORD *)(v26 + 8);
            v29 = *(_DWORD *)v26;
            v26 += 16;
            sub_B99FD0(v11, v29, v28);
          }
        }
        return v11;
      }
    case 6:
    case 7:
    case 8:
    case 9:
      if ( !a6 )
        goto LABEL_11;
      v35 = (unsigned int)sub_F6F100(a2);
      v36 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64))(**(_QWORD **)(a1 + 80) + 56LL))(
              *(_QWORD *)(a1 + 80),
              v35,
              a3,
              a4);
      if ( !v36 )
      {
        v72 = 257;
        v41 = sub_BD2C40(72, unk_3F10FD0);
        v42 = v41;
        if ( v41 )
        {
          v43 = *(_QWORD ***)(a3 + 8);
          v64 = (__int64)v41;
          v44 = *((unsigned __int8 *)v43 + 8);
          if ( (unsigned int)(v44 - 17) > 1 )
          {
            v46 = sub_BCB2A0(*v43);
          }
          else
          {
            BYTE4(v70) = (_BYTE)v44 == 18;
            LODWORD(v70) = *((_DWORD *)v43 + 8);
            v45 = (__int64 *)sub_BCB2A0(*v43);
            v46 = sub_BCE1B0(v45, v70);
          }
          sub_B523C0(v64, v46, 53, v35, a3, a4, (__int64)v71, 0, 0, 0);
          v42 = (_QWORD *)v64;
        }
        v65 = (__int64)v42;
        (*(void (__fastcall **)(_QWORD, _QWORD *, __int64, _QWORD, _QWORD))(**(_QWORD **)(a1 + 88) + 16LL))(
          *(_QWORD *)(a1 + 88),
          v42,
          a5,
          *(_QWORD *)(a1 + 56),
          *(_QWORD *)(a1 + 64));
        v36 = v65;
        v47 = *(_QWORD *)a1;
        v61 = *(_QWORD *)a1 + 16LL * *(unsigned int *)(a1 + 8);
        if ( *(_QWORD *)a1 != v61 )
        {
          do
          {
            v48 = *(_QWORD *)(v47 + 8);
            v49 = *(_DWORD *)v47;
            v47 += 16;
            v66 = v36;
            sub_B99FD0(v36, v49, v48);
            v36 = v66;
          }
          while ( v61 != v47 );
        }
      }
      v37 = a5;
      v38 = a4;
      v39 = a3;
      v40 = v36;
      return sub_B36550((unsigned int **)a1, v40, v39, v38, v37, 0);
    case 12:
    case 13:
    case 14:
    case 15:
LABEL_11:
      v18 = sub_F6F040(a2);
      v72 = 257;
      return sub_B33C40(a1, v18, a3, a4, (unsigned int)v70, (__int64)v71);
    default:
      BUG();
  }
}
