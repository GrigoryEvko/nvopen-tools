// Function: sub_21C1860
// Address: 0x21c1860
//
__int64 __fastcall sub_21C1860(__int64 a1, __int64 a2)
{
  __int64 v2; // rbp
  __int64 v3; // r12
  __int64 v4; // r14
  __int16 v5; // kr28_2
  __int64 result; // rax
  __int64 v8; // r9
  __int64 v9; // rax
  __int64 v10; // rbx
  int v11; // ecx
  const __m128i *v12; // r15
  __int64 v13; // rdx
  const __m128i *v14; // rbx
  unsigned __int64 v15; // r8
  __m128i *v16; // rax
  const __m128i *v17; // rbx
  __int64 v18; // rax
  __int64 v19; // rsi
  _QWORD *v20; // r15
  __int64 *v21; // r10
  unsigned int v22; // eax
  __int64 v23; // rcx
  int v24; // r8d
  __int64 v25; // r11
  __int64 v26; // r15
  __int64 v27; // rdx
  __int64 v28; // rcx
  __int64 v29; // r8
  __int64 v30; // r9
  __int64 v31; // [rsp-100h] [rbp-100h]
  unsigned __int64 v32; // [rsp-F8h] [rbp-F8h]
  unsigned __int64 v33; // [rsp-F8h] [rbp-F8h]
  __int64 v34; // [rsp-F0h] [rbp-F0h]
  int v35; // [rsp-E8h] [rbp-E8h]
  unsigned int v36; // [rsp-E8h] [rbp-E8h]
  unsigned int v37; // [rsp-E0h] [rbp-E0h]
  unsigned int v38; // [rsp-E0h] [rbp-E0h]
  __int64 v39; // [rsp-D8h] [rbp-D8h] BYREF
  int v40; // [rsp-D0h] [rbp-D0h]
  __int64 *v41; // [rsp-C8h] [rbp-C8h] BYREF
  __int64 v42; // [rsp-C0h] [rbp-C0h]
  _QWORD v43[23]; // [rsp-B8h] [rbp-B8h] BYREF

  v5 = *(_WORD *)(a2 + 24) - 858;
  v43[22] = v2;
  v43[20] = v4;
  v43[18] = v3;
  switch ( v5 )
  {
    case 0:
      v8 = 3720;
      goto LABEL_4;
    case 1:
      v8 = 3711;
      goto LABEL_4;
    case 2:
      v8 = 3714;
      goto LABEL_4;
    case 3:
      v8 = 3717;
      goto LABEL_4;
    case 4:
      v8 = 3732;
      goto LABEL_4;
    case 5:
      v8 = 3723;
      goto LABEL_4;
    case 6:
      v8 = 3726;
      goto LABEL_4;
    case 7:
      v8 = 3729;
      goto LABEL_4;
    case 8:
      v8 = 3741;
      goto LABEL_4;
    case 9:
      v8 = 3735;
      goto LABEL_4;
    case 10:
      v8 = 3738;
      goto LABEL_4;
    case 11:
    case 12:
    case 13:
    case 14:
    case 15:
    case 16:
    case 17:
    case 18:
    case 19:
    case 20:
    case 21:
      return 0;
    case 22:
      v8 = 3654;
      goto LABEL_4;
    case 23:
      v8 = 3645;
      goto LABEL_4;
    case 24:
      v8 = 3648;
      goto LABEL_4;
    case 25:
      v8 = 3651;
      goto LABEL_4;
    case 26:
      v8 = 3666;
      goto LABEL_4;
    case 27:
      v8 = 3657;
      goto LABEL_4;
    case 28:
      v8 = 3660;
      goto LABEL_4;
    case 29:
      v8 = 3663;
      goto LABEL_4;
    case 30:
      v8 = 3675;
      goto LABEL_4;
    case 31:
      v8 = 3669;
      goto LABEL_4;
    case 32:
      v8 = 3672;
      goto LABEL_4;
    case 33:
      v8 = 3786;
      goto LABEL_4;
    case 34:
      v8 = 3777;
      goto LABEL_4;
    case 35:
      v8 = 3780;
      goto LABEL_4;
    case 36:
      v8 = 3783;
      goto LABEL_4;
    case 37:
      v8 = 3798;
      goto LABEL_4;
    case 38:
      v8 = 3789;
      goto LABEL_4;
    case 39:
      v8 = 3792;
      goto LABEL_4;
    case 40:
      v8 = 3795;
      goto LABEL_4;
    case 41:
      v8 = 3807;
      goto LABEL_4;
    case 42:
      v8 = 3801;
      goto LABEL_4;
    case 43:
      v8 = 3804;
      goto LABEL_4;
    case 44:
      v8 = 3753;
      goto LABEL_4;
    case 45:
      v8 = 3744;
      goto LABEL_4;
    case 46:
      v8 = 3747;
      goto LABEL_4;
    case 47:
      v8 = 3750;
      goto LABEL_4;
    case 48:
      v8 = 3765;
      goto LABEL_4;
    case 49:
      v8 = 3756;
      goto LABEL_4;
    case 50:
      v8 = 3759;
      goto LABEL_4;
    case 51:
      v8 = 3762;
      goto LABEL_4;
    case 52:
      v8 = 3774;
      goto LABEL_4;
    case 53:
      v8 = 3768;
      goto LABEL_4;
    case 54:
      v8 = 3771;
      goto LABEL_4;
    case 55:
      v8 = 3819;
      goto LABEL_4;
    case 56:
      v8 = 3810;
      goto LABEL_4;
    case 57:
      v8 = 3813;
      goto LABEL_4;
    case 58:
      v8 = 3816;
      goto LABEL_4;
    case 59:
      v8 = 3831;
      goto LABEL_4;
    case 60:
      v8 = 3822;
      goto LABEL_4;
    case 61:
      v8 = 3825;
      goto LABEL_4;
    case 62:
      v8 = 3828;
      goto LABEL_4;
    case 63:
      v8 = 3840;
      goto LABEL_4;
    case 64:
      v8 = 3834;
      goto LABEL_4;
    case 65:
      v8 = 3837;
      goto LABEL_4;
    case 66:
      v8 = 3721;
      goto LABEL_4;
    case 67:
      v8 = 3712;
      goto LABEL_4;
    case 68:
      v8 = 3715;
      goto LABEL_4;
    case 69:
      v8 = 3718;
      goto LABEL_4;
    case 70:
      v8 = 3733;
      goto LABEL_4;
    case 71:
      v8 = 3724;
      goto LABEL_4;
    case 72:
      v8 = 3727;
      goto LABEL_4;
    case 73:
      v8 = 3730;
      goto LABEL_4;
    case 74:
      v8 = 3742;
      goto LABEL_4;
    case 75:
      v8 = 3736;
      goto LABEL_4;
    case 76:
      v8 = 3739;
      goto LABEL_4;
    case 77:
      v8 = 3688;
      goto LABEL_4;
    case 78:
      v8 = 3679;
      goto LABEL_4;
    case 79:
      v8 = 3682;
      goto LABEL_4;
    case 80:
      v8 = 3685;
      goto LABEL_4;
    case 81:
      v8 = 3700;
      goto LABEL_4;
    case 82:
      v8 = 3691;
      goto LABEL_4;
    case 83:
      v8 = 3694;
      goto LABEL_4;
    case 84:
      v8 = 3697;
      goto LABEL_4;
    case 85:
      v8 = 3709;
      goto LABEL_4;
    case 86:
      v8 = 3703;
      goto LABEL_4;
    case 87:
      v8 = 3706;
      goto LABEL_4;
    case 88:
      v8 = 3655;
      goto LABEL_4;
    case 89:
      v8 = 3646;
      goto LABEL_4;
    case 90:
      v8 = 3649;
      goto LABEL_4;
    case 91:
      v8 = 3652;
      goto LABEL_4;
    case 92:
      v8 = 3667;
      goto LABEL_4;
    case 93:
      v8 = 3658;
      goto LABEL_4;
    case 94:
      v8 = 3661;
      goto LABEL_4;
    case 95:
      v8 = 3664;
      goto LABEL_4;
    case 96:
      v8 = 3676;
      goto LABEL_4;
    case 97:
      v8 = 3670;
      goto LABEL_4;
    case 98:
      v8 = 3673;
      goto LABEL_4;
    case 99:
      v8 = 3787;
      goto LABEL_4;
    case 100:
      v8 = 3778;
      goto LABEL_4;
    case 101:
      v8 = 3781;
      goto LABEL_4;
    case 102:
      v8 = 3784;
      goto LABEL_4;
    case 103:
      v8 = 3799;
      goto LABEL_4;
    case 104:
      v8 = 3790;
      goto LABEL_4;
    case 105:
      v8 = 3793;
      goto LABEL_4;
    case 106:
      v8 = 3796;
      goto LABEL_4;
    case 107:
      v8 = 3808;
      goto LABEL_4;
    case 108:
      v8 = 3802;
      goto LABEL_4;
    case 109:
      v8 = 3805;
      goto LABEL_4;
    case 110:
      v8 = 3754;
      goto LABEL_4;
    case 111:
      v8 = 3745;
      goto LABEL_4;
    case 112:
      v8 = 3748;
      goto LABEL_4;
    case 113:
      v8 = 3751;
      goto LABEL_4;
    case 114:
      v8 = 3766;
      goto LABEL_4;
    case 115:
      v8 = 3757;
      goto LABEL_4;
    case 116:
      v8 = 3760;
      goto LABEL_4;
    case 117:
      v8 = 3763;
      goto LABEL_4;
    case 118:
      v8 = 3775;
      goto LABEL_4;
    case 119:
      v8 = 3769;
      goto LABEL_4;
    case 120:
      v8 = 3772;
      goto LABEL_4;
    case 121:
      v8 = 3820;
      goto LABEL_4;
    case 122:
      v8 = 3811;
      goto LABEL_4;
    case 123:
      v8 = 3814;
      goto LABEL_4;
    case 124:
      v8 = 3817;
      goto LABEL_4;
    case 125:
      v8 = 3832;
      goto LABEL_4;
    case 126:
      v8 = 3823;
      goto LABEL_4;
    case 127:
      v8 = 3826;
      goto LABEL_4;
    case 128:
      v8 = 3829;
      goto LABEL_4;
    case 129:
      v8 = 3841;
      goto LABEL_4;
    case 130:
      v8 = 3835;
      goto LABEL_4;
    case 131:
      v8 = 3838;
      goto LABEL_4;
    case 132:
      v8 = 3722;
      goto LABEL_4;
    case 133:
      v8 = 3713;
      goto LABEL_4;
    case 134:
      v8 = 3716;
      goto LABEL_4;
    case 135:
      v8 = 3719;
      goto LABEL_4;
    case 136:
      v8 = 3734;
      goto LABEL_4;
    case 137:
      v8 = 3725;
      goto LABEL_4;
    case 138:
      v8 = 3728;
      goto LABEL_4;
    case 139:
      v8 = 3731;
      goto LABEL_4;
    case 140:
      v8 = 3743;
      goto LABEL_4;
    case 141:
      v8 = 3737;
      goto LABEL_4;
    case 142:
      v8 = 3740;
      goto LABEL_4;
    case 143:
      v8 = 3689;
      goto LABEL_4;
    case 144:
      v8 = 3680;
      goto LABEL_4;
    case 145:
      v8 = 3683;
      goto LABEL_4;
    case 146:
      v8 = 3686;
      goto LABEL_4;
    case 147:
      v8 = 3701;
      goto LABEL_4;
    case 148:
      v8 = 3692;
      goto LABEL_4;
    case 149:
      v8 = 3695;
      goto LABEL_4;
    case 150:
      v8 = 3698;
      goto LABEL_4;
    case 151:
      v8 = 3710;
      goto LABEL_4;
    case 152:
      v8 = 3704;
      goto LABEL_4;
    case 153:
      v8 = 3707;
      goto LABEL_4;
    case 154:
      v8 = 3656;
      goto LABEL_4;
    case 155:
      v8 = 3647;
      goto LABEL_4;
    case 156:
      v8 = 3650;
      goto LABEL_4;
    case 157:
      v8 = 3653;
      goto LABEL_4;
    case 158:
      v8 = 3668;
      goto LABEL_4;
    case 159:
      v8 = 3659;
      goto LABEL_4;
    case 160:
      v8 = 3662;
      goto LABEL_4;
    case 161:
      v8 = 3665;
      goto LABEL_4;
    case 162:
      v8 = 3677;
      goto LABEL_4;
    case 163:
      v8 = 3671;
      goto LABEL_4;
    case 164:
      v8 = 3674;
      goto LABEL_4;
    case 165:
      v8 = 3788;
      goto LABEL_4;
    case 166:
      v8 = 3779;
      goto LABEL_4;
    case 167:
      v8 = 3782;
      goto LABEL_4;
    case 168:
      v8 = 3785;
      goto LABEL_4;
    case 169:
      v8 = 3800;
      goto LABEL_4;
    case 170:
      v8 = 3791;
      goto LABEL_4;
    case 171:
      v8 = 3794;
      goto LABEL_4;
    case 172:
      v8 = 3797;
      goto LABEL_4;
    case 173:
      v8 = 3809;
      goto LABEL_4;
    case 174:
      v8 = 3803;
      goto LABEL_4;
    case 175:
      v8 = 3806;
      goto LABEL_4;
    case 176:
      v8 = 3755;
      goto LABEL_4;
    case 177:
      v8 = 3746;
      goto LABEL_4;
    case 178:
      v8 = 3749;
      goto LABEL_4;
    case 179:
      v8 = 3752;
      goto LABEL_4;
    case 180:
      v8 = 3767;
      goto LABEL_4;
    case 181:
      v8 = 3758;
      goto LABEL_4;
    case 182:
      v8 = 3761;
      goto LABEL_4;
    case 183:
      v8 = 3764;
      goto LABEL_4;
    case 184:
      v8 = 3776;
      goto LABEL_4;
    case 185:
      v8 = 3770;
      goto LABEL_4;
    case 186:
      v8 = 3773;
      goto LABEL_4;
    case 187:
      v8 = 3821;
      goto LABEL_4;
    case 188:
      v8 = 3812;
      goto LABEL_4;
    case 189:
      v8 = 3815;
      goto LABEL_4;
    case 190:
      v8 = 3818;
      goto LABEL_4;
    case 191:
      v8 = 3833;
      goto LABEL_4;
    case 192:
      v8 = 3824;
      goto LABEL_4;
    case 193:
      v8 = 3827;
      goto LABEL_4;
    case 194:
      v8 = 3830;
      goto LABEL_4;
    case 195:
      v8 = 3842;
      goto LABEL_4;
    case 196:
      v8 = 3836;
      goto LABEL_4;
    case 197:
      v8 = 3839;
LABEL_4:
      v9 = *(unsigned int *)(a2 + 56);
      v10 = *(_QWORD *)(a2 + 32);
      v11 = 0;
      v41 = v43;
      v42 = 0x800000000LL;
      v12 = (const __m128i *)(v10 + 40 * v9);
      v13 = 40 * v9 - 40;
      v14 = (const __m128i *)(v10 + 40);
      v15 = 0xCCCCCCCCCCCCCCCDLL * (v13 >> 3);
      v16 = (__m128i *)v43;
      if ( (unsigned __int64)v13 > 0x140 )
      {
        v36 = v8;
        v33 = 0xCCCCCCCCCCCCCCCDLL * (v13 >> 3);
        sub_16CD150((__int64)&v41, v43, v33, 16, v15, v8);
        v11 = v42;
        v8 = v36;
        LODWORD(v15) = v33;
        v16 = (__m128i *)&v41[2 * (unsigned int)v42];
      }
      if ( v14 != v12 )
      {
        do
        {
          if ( v16 )
            *v16 = _mm_loadu_si128(v14);
          v14 = (const __m128i *)((char *)v14 + 40);
          ++v16;
        }
        while ( v12 != v14 );
        v11 = v42;
      }
      v17 = *(const __m128i **)(a2 + 32);
      LODWORD(v42) = v15 + v11;
      v18 = (unsigned int)(v15 + v11);
      if ( HIDWORD(v42) <= (unsigned int)(v15 + v11) )
      {
        v38 = v8;
        sub_16CD150((__int64)&v41, v43, 0, 16, v15, v8);
        v18 = (unsigned int)v42;
        v8 = v38;
      }
      *(__m128i *)&v41[2 * v18] = _mm_loadu_si128(v17);
      v19 = *(_QWORD *)(a2 + 72);
      v20 = *(_QWORD **)(a1 + 272);
      v21 = v41;
      v22 = v42 + 1;
      v39 = v19;
      v23 = *(_QWORD *)(a2 + 40);
      LODWORD(v42) = v22;
      v24 = *(_DWORD *)(a2 + 60);
      v25 = v22;
      if ( v19 )
      {
        v31 = v23;
        v35 = *(_DWORD *)(a2 + 60);
        v32 = (unsigned __int64)v41;
        v34 = v22;
        v37 = v8;
        sub_1623A60((__int64)&v39, v19, 2);
        v23 = v31;
        v24 = v35;
        v21 = (__int64 *)v32;
        v25 = v34;
        v8 = v37;
      }
      v40 = *(_DWORD *)(a2 + 64);
      v26 = sub_1D23DE0(v20, v8, (__int64)&v39, v23, v24, v8, v21, v25);
      sub_1D444E0(*(_QWORD *)(a1 + 272), a2, v26);
      sub_1D49010(v26);
      sub_1D2DC70(*(const __m128i **)(a1 + 272), a2, v27, v28, v29, v30);
      if ( v39 )
        sub_161E7C0((__int64)&v39, v39);
      if ( v41 != v43 )
        _libc_free((unsigned __int64)v41);
      result = 1;
      break;
    default:
      result = 0;
      break;
  }
  return result;
}
